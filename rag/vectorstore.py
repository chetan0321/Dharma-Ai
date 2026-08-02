"""
Vector store wrapper. ChromaDB persistent.
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import Chroma

from rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "indian_legal")
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )
