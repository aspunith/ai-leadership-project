"""
generator.py — Synthesise a leadership-ready answer from retrieved context.

If an OpenAI key is available, uses GPT-4o to compose the answer.
Otherwise, falls back to returning the retrieved passages directly.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

import config

# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI Leadership Insight Agent for ACME Corporation.
Your role is to answer questions from senior leadership using ONLY the
company documents provided as context.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information, say so clearly.
3. Be concise, factual, and use bullet points where appropriate.
4. Cite specific numbers, dates, or document sections when available.
5. Do NOT hallucinate or invent data.
"""


def _format_context(documents: List[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"--- Passage {i} (source: {source}) ---\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Public API ───────────────────────────────────────────────────────────

def generate_answer(query: str, documents: List[Document]) -> str:
    """
    Generate an answer to *query* grounded in *documents*.

    Uses OpenAI ChatCompletion when available; otherwise returns a
    formatted summary of the retrieved passages.
    """
    context = _format_context(documents)

    if config.USE_OPENAI:
        return _openai_answer(query, context)
    else:
        return _fallback_answer(query, context)


def _openai_answer(query: str, context: str) -> str:
    """Call OpenAI GPT to synthesise an answer."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_ANSWER_TOKENS,
        openai_api_key=config.OPENAI_API_KEY,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a clear, concise answer for leadership."
        )),
    ]

    response = llm.invoke(messages)
    return response.content


def _fallback_answer(query: str, context: str) -> str:
    """Return retrieved passages when no LLM is configured."""
    separator = "=" * 60
    header = (
        "[No OpenAI key configured — showing retrieved passages]\n"
        f"Question: {query}\n"
        f"{separator}\n"
    )
    return header + context
