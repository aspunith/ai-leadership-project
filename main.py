"""
main.py — CLI entry-point for the AI Leadership Insight Agent.

Usage:
    python main.py --ingest                         Ingest documents
    python main.py --query "your question here"     Ask a question
    python main.py --interactive                    Interactive Q&A loop
"""

from __future__ import annotations

import argparse
import sys

from agent.ingest import ingest_pipeline
from agent.retriever import retrieve
from agent.generator import generate_answer


def run_query(question: str) -> str:
    """Run the full RAG pipeline for a single question."""
    docs = retrieve(question)
    if not docs:
        return "No relevant information found in the ingested documents."
    answer = generate_answer(question, docs)
    return answer


def interactive_mode() -> None:
    """Run an interactive Q&A loop."""
    print("=" * 60)
    print("  AI Leadership Insight Agent — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    while True:
        try:
            question = input("\n📌 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        answer = run_query(question)
        print(f"\n{'─' * 60}")
        print(answer)
        print(f"{'─' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Leadership Insight Agent"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest company documents into the vector store.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Ask a single question.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive Q&A session.",
    )

    args = parser.parse_args()

    if not any([args.ingest, args.query, args.interactive]):
        parser.print_help()
        sys.exit(1)

    if args.ingest:
        print("[main] Starting document ingestion …")
        ingest_pipeline()

    if args.query:
        answer = run_query(args.query)
        print(f"\n{'=' * 60}")
        print(answer)
        print(f"{'=' * 60}")

    if args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
