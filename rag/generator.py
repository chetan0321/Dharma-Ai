"""
Generator — calls Llama 3.3 70B via Groq with a strict grounding prompt.
"""

import os
from groq import Groq


SYSTEM_PROMPT = """You are Dharma.AI, a research assistant specializing in Indian law.

RULES:
1. Answer ONLY based on the retrieved context below. If the context doesn't contain the answer, say "I don't have enough information in my knowledge base to answer this."
2. Always cite the source document name and section number in your answer (e.g., "Source: IPC Section 379").
3. Never invent case law, section numbers, or legal facts.
4. If a question is out of scope (not about Indian law), politely decline.
5. Be concise. Use plain English. Use bullet points for clarity.

Format your answer as:
- **Direct answer** (1-3 sentences)
- **Citation** (source + section)
- **Relevant excerpts** (1-2 short quotes from context)
"""


class AnswerGenerator:
    """Calls Groq API with Llama 3.3 70B."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 800,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, context_docs) -> dict:
        """Generate answer given question and retrieved docs."""
        # Format context
        context = self._format_context(context_docs)

        user_prompt = f"""Retrieved context:

{context}

---

Question: {question}

Answer (with citations):"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "answer": response.choices[0].message.content,
            "tokens": response.usage.total_tokens if response.usage else 0,
        }

    @staticmethod
    def _format_context(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "N/A")
            parts.append(
                f"[{i}] Source: {source} | Section: {section}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
