"""
Embeddings wrapper. Currently a thin layer over sentence-transformers
so we can swap it for OpenAI/Cohere/finetuned later.
"""

import os
from functools import lru_cache

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _EmbeddingClass = HuggingFaceEmbeddings
except ImportError:
    # Fallback to community package if langchain-huggingface not installed
    from langchain_community.embeddings import SentenceTransformerEmbeddings as HuggingFaceEmbeddings  # type: ignore
    _EmbeddingClass = HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings():
    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return _EmbeddingClass(model_name=model_name)
