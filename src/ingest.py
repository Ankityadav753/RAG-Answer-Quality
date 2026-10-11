"""
Document ingestion pipeline for DAI-009: RAG Answer Quality Benchmark.

Loads raw policy documents, splits them into deterministic chunks,
and persists them idempotently into the SQLite database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is available in sys.path for direct execution
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Internal dependencies
from db.database import DEFAULT_DB_PATH, get_db_connection, init_db
from src.chunker import chunk_text

DEFAULT_DOCS_PATH: Path = PROJECT_ROOT / "data" / "sample_documents.json"


def load_documents_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and parse the JSON document corpus."""
    if not file_path.exists():
        raise FileNotFoundError(f"Document corpus file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Corpus must be a JSON array of documents, got {type(data).__name__}")
    return data


def ingest_documents(
    docs_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
) -> Dict[str, Any]:
    """
    Ingest documents into SQLite database idempotently.
    
    Args:
        docs_path: Path to sample_documents.json (defaults to data/sample_documents.json).
        db_path: Path to SQLite DB file (defaults to db/rag_bench.db).
        chunk_size: Maximum words per chunk.
        chunk_overlap: Word overlap between adjacent chunks.
        
    Returns:
        Dict[str, Any]: Ingestion summary metrics.
    """
    target_docs = Path(docs_path).resolve() if docs_path else DEFAULT_DOCS_PATH
    target_db = Path(db_path).resolve() if db_path else DEFAULT_DB_PATH

    # Ensure database schema is initialized if table doesn't exist yet
    init_db(target_db)

    documents = load_documents_file(target_docs)

    total_docs_ingested = 0
    total_chunks_ingested = 0

    with get_db_connection(target_db) as conn:
        cursor = conn.cursor()

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            doc_id = str(doc.get("id", "")).strip()
            title = str(doc.get("title", "")).strip()
            source = str(doc.get("source", "")).strip() if doc.get("source") else None
            content = str(doc.get("content", ""))

            if not doc_id or not title:
                # Skip invalid documents missing core primary identifiers
                continue

            # 1. Upsert document record
            cursor.execute(
                """
                INSERT INTO documents (id, title, source, content)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    content = excluded.content;
                """,
                (doc_id, title, source, content),
            )
            total_docs_ingested += 1

            # 2. Clear any prior chunks for this document to guarantee idempotency
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = ?;",
                (doc_id,),
            )

            # 3. Generate chunks if content is non-empty
            if content.strip():
                chunks = chunk_text(
                    text=content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                for chunk in chunks:
                    chunk_id = f"{doc_id}_chk_{chunk.chunk_index:02d}"
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content, token_count
                        ) VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            chunk_id,
                            doc_id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.token_count,
                        ),
                    )
                    total_chunks_ingested += 1

    return {
        "documents_ingested": total_docs_ingested,
        "chunks_ingested": total_chunks_ingested,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "db_path": str(target_db),
    }


def main() -> int:
    """CLI execution entrypoint for document ingestion."""
    print("[INFO] Starting document ingestion pipeline...")
    try:
        summary = ingest_documents()
        print(f"[SUCCESS] Ingestion completed:")
        print(f"  - Database Path      : {summary['db_path']}")
        print(f"  - Documents Ingested : {summary['documents_ingested']}")
        print(f"  - Chunks Ingested    : {summary['chunks_ingested']}")
        print(f"  - Chunk Size / Overlap: {summary['chunk_size']} / {summary['chunk_overlap']} words")
        return 0
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
