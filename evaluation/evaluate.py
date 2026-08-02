"""
Evaluation — runs both RAG and baseline on the test set, scores with LLM-as-judge.

Features:
- Retry with exponential backoff on Groq rate limit errors
- Incremental save after each question (crash-safe)
- Resume: skips already-scored questions if scores.json exists
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rag.pipeline import RAGPipeline
from rag.baseline import BaselineLLM

load_dotenv()

DATASET_PATH = Path("evaluation/eval_dataset.json")
OUTPUT_PATH = Path("evaluation/results/scores.json")

JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """You are an evaluation judge. Score the candidate answer for FAITHFULNESS to the expected answer on a 1-5 scale.

1 = Completely wrong or hallucinated
2 = Mostly wrong, some correct elements
3 = Partially correct
4 = Mostly correct, minor errors
5 = Fully faithful to expected answer

Question: {question}
Expected answer: {expected}
Candidate answer: {candidate}

Reply with ONLY a single number 1-5, nothing else."""


def judge(question: str, expected: str, candidate: str, client: Groq) -> int:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, candidate=candidate
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content.strip()
    # Parse first digit
    for c in text:
        if c.isdigit():
            return int(c)
    return 0


def call_with_retry(fn, *args, retries: int = 3, base_delay: float = 60.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on rate limit errors."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = base_delay * (2 ** attempt)
                print(f"  Rate limit — waiting {wait:.0f}s (retry {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"All {retries} retries exhausted due to rate limiting.")


def _save(results: list, final: bool = False) -> None:
    """Save results to OUTPUT_PATH (partial or final)."""
    n = len(results)
    if n == 0:
        return
    rag_avg = sum(r["rag_score"] for r in results) / n
    base_avg = sum(r["base_score"] for r in results) / n
    rag_lat = sum(r["rag_latency_s"] for r in results) / n
    base_lat = sum(r["base_latency_s"] for r in results) / n

    summary = {
        "num_questions": n,
        "complete": final,
        "rag_faithfulness_avg": round(rag_avg, 2),
        "baseline_faithfulness_avg": round(base_avg, 2),
        "rag_avg_latency_s": round(rag_lat, 2),
        "baseline_avg_latency_s": round(base_lat, 2),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if final:
        print("\n" + "=" * 60)
        print(f"[RAG]      Faithfulness avg: {rag_avg:.2f} / 5")
        print(f"[Baseline] Faithfulness avg: {base_avg:.2f} / 5")
        print(f"[RAG]      Avg latency:      {rag_lat:.2f}s")
        print(f"[Baseline] Avg latency:      {base_lat:.2f}s")
        print(f"\nResults saved to: {OUTPUT_PATH}")


def evaluate() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    with DATASET_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)

    rag = RAGPipeline()
    baseline = BaselineLLM()
    judge_client = Groq(api_key=api_key)

    # Resume: load existing partial results and skip completed questions
    existing_results: list = []
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open(encoding="utf-8") as f:
                prev = json.load(f)
                existing_results = prev.get("results", [])
                if prev.get("complete"):
                    print(f"Evaluation already complete ({len(existing_results)} questions). Delete")
                    print(f"{OUTPUT_PATH} to re-run.\n")
                    _save(existing_results, final=True)
                    return
        except Exception:
            pass

    completed_ids = {r["id"] for r in existing_results}
    results = list(existing_results)
    remaining = [item for item in dataset if item["id"] not in completed_ids]

    print(f"Evaluating {len(remaining)} questions "
          f"(skipping {len(completed_ids)} already scored)...\n")

    for item in remaining:
        q = item["question"]
        expected = item["expected_answer"]
        category = item.get("category", "general")

        # RAG answer
        t0 = time.time()
        rag_out = call_with_retry(rag.answer, q)
        rag_time = time.time() - t0
        rag_answer = rag_out["answer"]

        # Baseline answer
        t0 = time.time()
        base_answer = call_with_retry(baseline.answer, q)
        base_time = time.time() - t0

        # Judge
        rag_score = call_with_retry(judge, q, expected, rag_answer, judge_client)
        base_score = call_with_retry(judge, q, expected, base_answer, judge_client)

        results.append({
            "id": item["id"],
            "category": category,
            "question": q,
            "expected": expected,
            "rag_answer": rag_answer,
            "base_answer": base_answer,
            "rag_score": rag_score,
            "base_score": base_score,
            "rag_latency_s": round(rag_time, 2),
            "base_latency_s": round(base_time, 2),
        })

        print(
            f"[{item['id']:>2}] {category:<14} | "
            f"RAG: {rag_score}/5 ({rag_time:.1f}s) | "
            f"Baseline: {base_score}/5 ({base_time:.1f}s)"
        )

        # Incremental save — crash-safe, resumable
        _save(results, final=False)

    # Final save with complete=True
    _save(results, final=True)


if __name__ == "__main__":
    evaluate()
