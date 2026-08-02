"""
Retriever — wraps ChromaDB with MMR search.
"""

import os

from langchain_community.vectorstores import Chroma
from langchain.schema import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _EmbedCls
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings as _EmbedCls  # type: ignore


class LegalRetriever:
    """Retrieves relevant chunks from Indian legal documents."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        k: int = 5,
        fetch_k: int = 20,
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "indian_legal")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        self.embeddings = _EmbedCls(model_name=model_name)
        self.vs = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )

    def retrieve(self, query: str) -> list[Document]:
        # Use invoke() — get_relevant_documents() is deprecated in LangChain 0.2+
        return self.retriever.invoke(query)
