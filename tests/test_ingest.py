"""
Integration tests for document ingestion and database idempotency (DAI-009).

Tests:
1. Ingesting sample_documents.json populates documents and document_chunks.
2. Ingestion is idempotent: running multiple times preserves exact record counts without duplicates.
3. Empty documents are ingested safely without throwing errors.
4. Foreign key integrity between chunks and documents is preserved.
5. Invalid chunking parameters trigger appropriate exceptions.

Compatible with both `python -m unittest` and `pytest`.
"""

import json
import tempfile
import unittest
from pathlib import Path

from db.database import get_db_connection, init_db
from src.ingest import DEFAULT_DOCS_PATH, ingest_documents


class TestIngestionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        """Create a clean temporary database for isolated ingestion testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_ingest.db"
        init_db(self.test_db_path)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_ingest_sample_documents(self) -> None:
        """Verify that sample_documents.json is ingested into documents and document_chunks."""
        summary = ingest_documents(
            docs_path=DEFAULT_DOCS_PATH,
            db_path=self.test_db_path,
            chunk_size=80,
            chunk_overlap=20,
        )

        self.assertEqual(summary["documents_ingested"], 11)
        self.assertGreater(summary["chunks_ingested"], 11)

        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()

            # Verify documents count
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents;")
            doc_cnt = cursor.fetchone()["cnt"]
            self.assertEqual(doc_cnt, 11)

            # Verify chunks count
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks;")
            chk_cnt = cursor.fetchone()["cnt"]
            self.assertEqual(chk_cnt, summary["chunks_ingested"])

            # Verify chunk indices start at 0
            cursor.execute("SELECT MIN(chunk_index) AS min_idx FROM document_chunks;")
            min_idx = cursor.fetchone()["min_idx"]
            self.assertEqual(min_idx, 0)

            # Verify all chunks reference valid documents
            cursor.execute(
                """
                SELECT COUNT(*) AS orphaned
                FROM document_chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE d.id IS NULL;
                """
            )
            orphaned = cursor.fetchone()["orphaned"]
            self.assertEqual(orphaned, 0)

    def test_ingestion_is_idempotent(self) -> None:
        """Verify that running ingestion repeatedly yields the exact same counts without duplication."""
        # Run 1
        summary_1 = ingest_documents(
            docs_path=DEFAULT_DOCS_PATH,
            db_path=self.test_db_path,
            chunk_size=80,
            chunk_overlap=20,
        )

        # Run 2 on same database
        summary_2 = ingest_documents(
            docs_path=DEFAULT_DOCS_PATH,
            db_path=self.test_db_path,
            chunk_size=80,
            chunk_overlap=20,
        )

        self.assertEqual(summary_1["documents_ingested"], summary_2["documents_ingested"])
        self.assertEqual(summary_1["chunks_ingested"], summary_2["chunks_ingested"])

        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents;")
            self.assertEqual(cursor.fetchone()["cnt"], 11)

            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks;")
            self.assertEqual(cursor.fetchone()["cnt"], summary_1["chunks_ingested"])

    def test_empty_document_handling(self) -> None:
        """Verify that a document with empty content is safely recorded with zero chunks."""
        custom_docs = [
            {
                "id": "DOC-EMPTY-01",
                "title": "Placeholder Policy",
                "category": "General",
                "source": "placeholder.md",
                "content": "   ",
            }
        ]
        temp_docs_file = Path(self.temp_dir.name) / "empty_doc.json"
        with open(temp_docs_file, "w", encoding="utf-8") as f:
            json.dump(custom_docs, f)

        summary = ingest_documents(
            docs_path=temp_docs_file,
            db_path=self.test_db_path,
            chunk_size=50,
            chunk_overlap=10,
        )

        self.assertEqual(summary["documents_ingested"], 1)
        self.assertEqual(summary["chunks_ingested"], 0)

        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents WHERE id = 'DOC-EMPTY-01';")
            self.assertEqual(cursor.fetchone()["cnt"], 1)

            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM document_chunks WHERE document_id = 'DOC-EMPTY-01';"
            )
            self.assertEqual(cursor.fetchone()["cnt"], 0)

    def test_invalid_parameters_raise_error(self) -> None:
        """Verify that invalid chunk parameters raise ValueError during ingestion."""
        with self.assertRaises(ValueError):
            ingest_documents(
                docs_path=DEFAULT_DOCS_PATH,
                db_path=self.test_db_path,
                chunk_size=0,
                chunk_overlap=0,
            )


if __name__ == "__main__":
    unittest.main()
