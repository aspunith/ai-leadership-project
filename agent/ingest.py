"""
ingest.py — Document loading, chunking, and vector-store creation.

Reads every .txt, .pdf, and .docx file from the company_documents/ folder,
splits them into overlapping chunks, embeds them, and persists them to a
ChromaDB vector store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

import config


# ── Loaders for each file type ───────────────────────────────────────────

def _build_loaders(docs_dir: str | Path) -> List[DirectoryLoader]:
    """Return one DirectoryLoader per supported file extension."""
    docs_dir = str(docs_dir)
    return [
        DirectoryLoader(
            docs_dir, glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        ),
        DirectoryLoader(
            docs_dir, glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        ),
        DirectoryLoader(
            docs_dir, glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
        ),
    ]


# ── Public API ───────────────────────────────────────────────────────────

def load_documents(docs_dir: str | Path | None = None) -> List[Document]:
    """Load all supported documents from *docs_dir*."""
    docs_dir = Path(docs_dir or config.DOCUMENTS_DIR)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents: List[Document] = []
    for loader in _build_loaders(docs_dir):
        try:
            documents.extend(loader.load())
        except Exception:
            # Silently skip loaders that find no matching files
            pass

    if not documents:
        raise ValueError(f"No documents found in {docs_dir}")

    print(f"[ingest] Loaded {len(documents)} document(s) from {docs_dir}")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunk(s)")
    return chunks


def get_embedding_function():
    """Return the appropriate embedding function based on config."""
    if config.USE_OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )


def build_vector_store(
    chunks: List[Document],
    persist_dir: str | Path | None = None,
):
    """Embed *chunks* and persist to a ChromaDB vector store."""
    import shutil
    from langchain_chroma import Chroma

    persist_dir = str(persist_dir or config.CHROMA_PERSIST_DIR)

    # Clear old vector store to avoid duplicates on re-ingest
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
        print(f"[ingest] Cleared old vector store at {persist_dir}")

    embeddings = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"[ingest] Vector store persisted to {persist_dir}")
    return vector_store


def ingest_pipeline(docs_dir: str | Path | None = None) -> None:
    """Run the full ingest pipeline: load → chunk → embed → persist."""
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("[ingest] Ingestion complete ✓")
