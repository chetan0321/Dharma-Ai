"""
RAG pipeline — orchestrates retrieval + generation.
"""

from typing import Optional

from rag.retriever import LegalRetriever
from rag.generator import AnswerGenerator


class RAGPipeline:
    """End-to-end RAG pipeline for Indian legal Q&A."""

    def __init__(
        self,
        retriever_k: int = 5,
        temperature: float = 0.1,
        model: str = "llama-3.3-70b-versatile",
    ):
        self.retriever = LegalRetriever(k=retriever_k)
        self.generator = AnswerGenerator(model=model, temperature=temperature)

    def answer(self, question: str) -> dict:
        """
        Answer a legal question with retrieval-augmented generation.

        Returns:
            {
                "answer": str,
                "sources": list[dict],
                "tokens": int,
            }
        """
        # 1. Retrieve
        docs = self.retriever.retrieve(question)

        # 2. Generate
        result = self.generator.generate(question, docs)

        # 3. Format sources for UI
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "unknown"),
                "section": doc.metadata.get("section", "N/A"),
                "content": doc.page_content,
            }
            for doc in docs
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "tokens": result.get("tokens", 0),
        }
