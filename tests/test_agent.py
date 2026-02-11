"""
tests/test_agent.py — Unit tests for the AI Leadership Insight Agent.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a small sample document."""
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "ACME Corp revenue for FY 2025 was $4.8 billion, "
        "a 14% increase year-over-year. Net income was $620 million. "
        "The Cloud Services division generated $1.92B in revenue. "
        "Engineering attrition rose to 17%.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Revenue was $4.8B. Net income $620M.",
            metadata={"source": "annual_report.txt"},
        ),
        Document(
            page_content="Engineering attrition 17%. Cloud Services grew 22%.",
            metadata={"source": "q3_report.txt"},
        ),
    ]


# ── Test: Document loading ───────────────────────────────────────────────

def test_load_documents(sample_docs_dir: Path):
    from agent.ingest import load_documents

    docs = load_documents(sample_docs_dir)
    assert len(docs) >= 1
    assert "ACME" in docs[0].page_content


def test_load_documents_missing_dir():
    from agent.ingest import load_documents

    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path")


# ── Test: Chunking ───────────────────────────────────────────────────────

def test_split_documents(sample_docs_dir: Path):
    from agent.ingest import load_documents, split_documents

    docs = load_documents(sample_docs_dir)
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200  # chunk_size + tolerance


# ── Test: Context formatting ─────────────────────────────────────────────

def test_format_context(sample_documents):
    from agent.generator import _format_context

    ctx = _format_context(sample_documents)
    assert "Passage 1" in ctx
    assert "Passage 2" in ctx
    assert "annual_report.txt" in ctx


# ── Test: Fallback answer (no OpenAI key) ────────────────────────────────

def test_fallback_answer(sample_documents):
    from agent.generator import _fallback_answer

    answer = _fallback_answer("What is our revenue?", "Revenue was $4.8B.")
    assert "Revenue was $4.8B" in answer
    assert "No OpenAI key" in answer


# ── Test: Full vector-store round-trip ───────────────────────────────────

def test_ingest_and_retrieve(sample_docs_dir: Path, tmp_path: Path):
    from agent.ingest import load_documents, split_documents, build_vector_store
    from agent.retriever import retrieve

    persist_dir = tmp_path / "test_vs"
    docs = load_documents(sample_docs_dir)
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)
    build_vector_store(chunks, persist_dir=persist_dir)

    results = retrieve(
        "What was the revenue?",
        persist_dir=persist_dir,
        top_k=2,
    )
    assert len(results) >= 1
    combined = " ".join(r.page_content for r in results)
    assert "4.8" in combined or "revenue" in combined.lower()
