# Dharma.AI — Project Plan

> RAG-based Q&A over Indian legal knowledge
> Owner: Chetan Reddy | Target: 3-4 days from start to polished GitHub repo
> Stack: Python · Streamlit · LangChain · ChromaDB · sentence-transformers · Groq (Llama 3.3 70B)

---

## Why this project (and not fine-tuning or classification)

1. **Job-market fit** — RAG is the most-requested skill on AI engineer job posts in Hyderabad right now. It's literally what Tachyon, V Solutions, Pype, SmartinfoLogiks are building.
2. **Skill fit** — you already know Streamlit, Groq, Llama 3.3 from Atlas.AI. You're 60% there.
3. **Buildable without a GPU** — runs on your laptop.
4. **Domain** — Indian legal/finance is a real underserved space. Recruiters notice when you pick a domain instead of doing "yet another movie recommender."
5. **Differentiator** — most students copy RAG tutorials. Yours will have **evaluation**, **citation of sources**, **a real Indian dataset**, and a **comparison: with-RAG vs without-RAG** — that's interview-grade.

---

## The build — 4 days, ~4-6 hours/day

### Day 1: Data + ingestion
- Source: Indian Kanoon open data (`https://api.indiankanoon.org`) OR curated subset of public legal PDFs (Constitution of India, IPC, IT Act, GST Acts — all on `https://www.indiacode.nic.in`)
- Goal: 50-100 documents, mix of statutes and judgments
- Tasks:
  - Build `data/ingest.py` — downloads, cleans, chunks (500 tokens, 50 overlap)
  - Save to `data/processed/chunks.jsonl`
  - Commit: `feat: data ingestion pipeline`

### Day 2: Vector store + retrieval
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (free, local, fast)
- Vector store: ChromaDB (persistent, file-based)
- Tasks:
  - Build `rag/embeddings.py` — wraps sentence-transformers
  - Build `rag/vectorstore.py` — ChromaDB ops (create, upsert, query)
  - Build `rag/retriever.py` — top-k retrieval with optional MMR for diversity
  - Test: 5 sample queries, print retrieved chunks
  - Commit: `feat: vector store and retriever`

### Day 3: LLM pipeline + Streamlit UI
- LLM: Llama 3.3 70B via Groq (you already have the API key from Atlas)
- Tasks:
  - Build `rag/generator.py` — prompt template (system + context + question)
  - Build `rag/pipeline.py` — full RAG chain (retrieve → prompt → generate)
  - Build `app.py` — Streamlit UI with:
    - Question input
    - "Answer with RAG" button
    - "Answer without RAG" button (baseline, for comparison)
    - Show retrieved chunks + source URLs
    - Show latency + token usage
  - Commit: `feat: RAG pipeline and Streamlit UI`

### Day 4: Evaluation + polish
- Build `evaluation/eval_dataset.json` — 20 Q&A pairs (mix of factual, ambiguous, out-of-scope)
- Build `evaluation/evaluate.py`:
  - Faithfulness: does the answer stick to retrieved context? (LLM-as-judge)
  - Relevance: is the retrieved context relevant? (cosine sim + LLM judge)
  - Latency p50, p95
  - Comparison: with-RAG vs without-RAG scores
- Polish:
  - Write killer README (architecture diagram, screenshots, eval results table)
  - Add `LICENSE` (MIT)
  - Add `.env.example`
  - Add `requirements.txt` pinned versions
  - Add a 60-second Loom video demo in README
- Commit: `feat: evaluation suite and README`

---

## Final file structure

```
dharma-ai/
├── README.md                    # The most important file — this is what recruiters read
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                       # Streamlit entry
├── data/
│   ├── raw/                     # Downloaded PDFs
│   ├── processed/chunks.jsonl   # Chunked data
│   └── scripts/ingest.py
├── rag/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── evaluation/
│   ├── eval_dataset.json
│   └── evaluate.py
└── docs/
    ├── architecture.md
    └── screenshots/
        ├── ui.png
        └── eval_results.png
```

---

## README.md outline (this is what gets you interviews)

```markdown
# Dharma.AI ⚖️
> RAG-based Q&A over Indian legal knowledge, built with Llama 3.3 + ChromaDB + Streamlit.

[Live Demo] · [Architecture] · [Eval Results]

## Why I built this
[2-3 sentences: Indian legal info is fragmented across 100s of acts, judgments, and government sites.
Lawyers, students, and citizens need a single place to ask plain-English questions and get cited answers.
Most existing chatbots hallucinate. This is a grounded alternative.]

## How it works
[Architecture diagram]

1. **Ingest** — Download 80+ Indian legal docs (Constitution, IPC, CrPC, IT Act, GST Acts) → chunk
2. **Embed** — sentence-transformers/all-MiniLM-L6-v2 → 384-dim vectors
3. **Store** — ChromaDB persistent store
4. **Retrieve** — top-k=5 with MMR reranking
5. **Generate** — Llama 3.3 70B via Groq with strict grounding prompt
6. **Cite** — every answer links back to source document + section

## Stack
- **LLM**: Llama 3.3 70B (Groq API)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **Orchestration**: LangChain
- **UI**: Streamlit
- **Eval**: Custom 20-question suite + LLM-as-judge

## Results
| Metric | With RAG | Without RAG |
|---|---|---|
| Faithfulness (1-5) | 4.6 | 2.1 |
| Answer relevance | 0.89 | 0.42 |
| Avg latency | 1.2s | 0.8s |
| Hallucination rate | 4% | 38% |

[chart/table here]

## Run it
[3 commands: clone, pip install, streamlit run]

## What I learned
- RAG is not just "throw docs in a DB" — chunking strategy and prompt design matter
- Llama 3.3 70B is fast enough on Groq for production-feel UX
- LLM-as-judge works for 80% of faithfulness checks; humans still needed for the rest

## Roadmap
- [ ] Add agentic tool-use (cite IPC section, fetch judgment text)
- [ ] Multilingual support (Hindi, Telugu)
- [ ] Fine-tune embeddings on Indian legal corpus
```

---

## Evaluation dataset — sample (you'll write 20 like this)

```json
[
  {
    "question": "What is the punishment for theft under IPC?",
    "expected_answer": "Section 379, imprisonment up to 3 years, or fine, or both.",
    "category": "factual",
    "source_doc": "IPC Section 379"
  },
  {
    "question": "Can a person be tried twice for the same offence?",
    "expected_answer": "No, Article 20(2) of the Constitution and Section 300 CrPC bar double jeopardy.",
    "category": "constitutional",
    "source_doc": "Constitution Article 20"
  },
  {
    "question": "What's the weather in Mumbai?",
    "expected_answer": "Out of scope — should be flagged as not answerable from legal corpus.",
    "category": "out_of_scope"
  }
]
```

---

## Code samples (skeleton)

### `rag/retriever.py`
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import MMRRetriever

class LegalRetriever:
    def __init__(self, persist_dir="./chroma_db", k=5):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20}
        )

    def retrieve(self, query: str):
        return self.retriever.get_relevant_documents(query)
```

### `rag/generator.py`
```python
from groq import Groq

SYSTEM_PROMPT = """You are Dharma.AI, a legal research assistant.
Answer ONLY based on the retrieved context. If the context doesn't contain the answer, say so.
Always cite the source document and section number in your answer.
Never invent case law or section numbers."""

class AnswerGenerator:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model = model

    def generate(self, question: str, context_chunks: list) -> dict:
        context = "\n\n---\n\n".join([c.page_content for c in context_chunks])
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return {
            "answer": resp.choices[0].message.content,
            "tokens": resp.usage.total_tokens,
            "latency_ms": resp.usage.total_time * 1000 if hasattr(resp.usage, 'total_time') else None
        }
```

### `evaluation/evaluate.py` (skeleton)
```python
import json
from rag.pipeline import RAGPipeline
from rag.baseline import BaselineLLM  # No retrieval

def evaluate():
    with open("evaluation/eval_dataset.json") as f:
        dataset = json.load(f)

    rag = RAGPipeline()
    baseline = BaselineLLM()

    results = {"with_rag": [], "without_rag": []}
    for item in dataset:
        rag_out = rag.answer(item["question"])
        base_out = baseline.answer(item["question"])
        results["with_rag"].append(rag_out)
        results["without_rag"].append(base_out)

    # Score with LLM-as-judge
    judge_scores = judge_results(results, dataset)

    print(json.dumps(judge_scores, indent=2))

def judge_results(results, dataset):
    # Use Llama 3.3 to score each (Q, expected, answer) for faithfulness
    # Return average scores per category
    ...
```

---

## Stretch goals (after Day 4, if you have time)

1. **Fine-tune a small LLM** (Phi-3 mini or Llama-3.2-1B) on Indian legal Q&A using QLoRA + unsloth on Colab. This is your Project 3 if you want to go deeper.
2. **Add agentic tool-use** — let the model decide whether to retrieve, fetch a specific section, or compare two sections.
3. **Multilingual** — add Hindi/Telugu support (Indian langs in sentence-transformers).
4. **Deploy** — put on Streamlit Community Cloud (free). Add the live link to your resume.

---

## Definition of done (so you know when to stop and apply)

- [ ] Repo is public, has 4+ meaningful commits, README is filled in
- [ ] Eval results table in README (with vs without RAG)
- [ ] 1-minute demo video or 3 screenshots
- [ ] Live demo URL (optional but powerful)
- [ ] Resume updated with this project
- [ ] LinkedIn post about the project (build-in-public)
- [ ] Cold email to 5-10 companies references this project
