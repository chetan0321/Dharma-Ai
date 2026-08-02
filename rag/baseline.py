"""
Baseline LLM — answers without RAG, for comparison.
"""

import os
from groq import Groq


SYSTEM_PROMPT = """You are a helpful legal research assistant. Answer the user's question about Indian law to the best of your general knowledge."""


class BaselineLLM:
    """Pure LLM baseline — no retrieval, no grounding."""

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

    def answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
