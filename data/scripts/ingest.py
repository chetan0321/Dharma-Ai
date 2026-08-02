"""
Data ingestion — load Indian legal documents, chunk, and push to ChromaDB.

Sources:
- indiacode.nic.in (Acts)
- indiankanoon.org (Judgments — API available)
- Static curated PDFs in data/raw/

Run:
    python -m data.scripts.ingest
"""

import os
import json
from pathlib import Path
from typing import Iterator

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _EmbedCls
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings as _EmbedCls  # type: ignore
from dotenv import load_dotenv

load_dotenv()

# Config
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "indian_legal")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdfs(raw_dir: Path) -> Iterator[Document]:
    """Yield LangChain Documents from all PDFs in raw_dir."""
    for pdf_path in raw_dir.glob("**/*.pdf"):
        print(f"Loading: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

            yield Document(
                page_content=full_text,
                metadata={
                    "source": pdf_path.name,
                    "section": "full_act",  # override per-document if you have a section index
                    "title": pdf_path.stem.replace("_", " ").title(),
                },
            )
        except Exception as e:
            print(f"  Failed to load {pdf_path}: {e}")


def chunk_docs(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def save_chunks(chunks: list[Document], out_path: Path) -> None:
    """Save chunks to JSONL for inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {"content": c.page_content, "metadata": c.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    embeddings = _EmbedCls(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    # vs.persist() removed — ChromaDB 0.4+ persists automatically
    return vs


def main():
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"Created {RAW_DIR}. Drop your PDFs in there and re-run.")
        return

    print("Step 1: Loading PDFs...")
    docs = list(load_pdfs(RAW_DIR))
    print(f"  Loaded {len(docs)} documents")

    if not docs:
        print("  No documents found. Add PDFs to data/raw/ and re-run.")
        return

    print("Step 2: Chunking...")
    chunks = chunk_docs(docs)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3: Saving chunks for inspection...")
    save_chunks(chunks, PROCESSED_DIR / "chunks.jsonl")

    print("Step 4: Building vector index...")
    build_index(chunks)
    print(f"  Index built and persisted to {CHROMA_DIR}")

    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
