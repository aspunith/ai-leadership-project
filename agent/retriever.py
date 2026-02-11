"""
retriever.py — Retrieve the most relevant document chunks for a query.

Loads the persisted ChromaDB vector store and performs a similarity search.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

import config
from agent.ingest import get_embedding_function


def get_retriever(
    persist_dir: str | Path | None = None,
    top_k: int = config.TOP_K_RESULTS,
):
    """Return a LangChain retriever backed by the persisted vector store."""
    from langchain_chroma import Chroma

    persist_dir = str(persist_dir or config.CHROMA_PERSIST_DIR)
    embeddings = get_embedding_function()

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve(
    query: str,
    persist_dir: str | Path | None = None,
    top_k: int = config.TOP_K_RESULTS,
) -> List[Document]:
    """Return the top-k most relevant chunks for *query*."""
    retriever = get_retriever(persist_dir=persist_dir, top_k=top_k)
    results = retriever.invoke(query)
    print(f"[retriever] Found {len(results)} relevant chunk(s)")
    return results
