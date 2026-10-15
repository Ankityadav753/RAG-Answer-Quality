"""
CLI interface for querying the RAG Answer Quality system (DAI-009).

Usage:
    python -m src.cli --question "What is the student Wi-Fi SSID?"
    python -m src.cli -q "What is the student Wi-Fi SSID?" --retriever dense
    python -m src.cli -q "Out of domain question" --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is available in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service import RAGResponse, RAGService


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Query the DAI-009 RAG Answer Quality system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Natural language question to ask the RAG system.",
    )
    parser.add_argument(
        "-r",
        "--retriever",
        type=str,
        choices=["lexical", "dense"],
        default="lexical",
        help="Retrieval configuration: 'lexical' (FTS5 BM25) or 'dense' (Cosine similarity).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["mock", "openai", "gemini"],
        default=None,
        help="LLM generator provider ('mock', 'openai', 'gemini'). Defaults to environment or 'mock'.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model identifier override for external providers (e.g. 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["mock", "openai", "sentence_transformers"],
        default=None,
        help="Embedding provider for dense retrieval.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom path to SQLite database file.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: human-readable 'text' or machine-readable 'json'.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Force re-ingestion of documents into the database before answering.",
    )
    return parser


def format_text_output(resp: RAGResponse) -> str:
    """Format RAGResponse into a structured human-readable terminal output."""
    lines: List[str] = [
        "=" * 72,
        "  DAI-009 RAG ANSWER QUALITY QUERY RESULT",
        "=" * 72,
        f"Question  : {resp.question}",
        f"Retriever : {resp.retriever_name} (top_k={resp.top_k})",
        f"Provider  : {resp.provider}",
        f"Latency   : {resp.latency_ms:.2f} ms",
        "-" * 72,
        "ANSWER:",
        f"{resp.answer}",
        "-" * 72,
        "STATUS & CITATIONS:",
        f"  Abstention Triggered : {'YES (Refused - No Evidence)' if resp.is_abstention else 'NO (Answer Generated)'}",
        f"  Source Documents     : {resp.source_doc_ids if resp.source_doc_ids else '[None]'}",
        f"  Source Chunks        : {resp.source_chunk_ids if resp.source_chunk_ids else '[None]'}",
        "-" * 72,
        f"RETRIEVED CONTEXT CHUNKS ({len(resp.retrieved_chunks)} total):",
    ]

    if not resp.retrieved_chunks:
        lines.append("  [No chunks retrieved]")
    else:
        for c in resp.retrieved_chunks:
            header = f"  [Rank {c.rank} | Score: {c.score:.4f} | Doc: {c.document_id} | Chunk: {c.chunk_id}]"
            snippet = f"  {c.content.strip()}"
            lines.append(header)
            lines.append(snippet)
            lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.question is not None:
        question = args.question.strip()
        if not question:
            print("[ERROR] Question cannot be empty.", file=sys.stderr)
            return 1
    else:
        # Check if piped input is available
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        else:
            try:
                question = input("Enter your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[INFO] Cancelled by user.")
                return 1

        if not question:
            print("[ERROR] Question cannot be empty.", file=sys.stderr)
            return 1

    # Apply CLI provider overrides to environment if passed
    if args.provider:
        os.environ["RAG_LLM_PROVIDER"] = args.provider
    if args.embedding_provider:
        os.environ["RAG_EMBEDDING_PROVIDER"] = args.embedding_provider
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    try:
        service = RAGService(
            db_path=args.db_path,
            retriever_type=args.retriever,
            top_k=args.top_k,
            auto_init=True,
        )

        if args.ingest:
            service.ingest(force_recompute_embeddings=True)

        response = service.query(question=question, top_k=args.top_k)

        if args.format == "json":
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(format_text_output(response))

        return 0

    except Exception as e:
        print(f"[ERROR] Query failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
