"""
Smoke tests for the SQLite database layer (DAI-009).

Tests schema initialization, foreign-key enforcement, cascading deletes,
and relational integrity across documents, chunks, runs, and evaluations.
Compatible with both `python -m unittest` and `pytest`.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from db.database import (
    get_connection,
    get_db_connection,
    get_db_path,
    get_table_names,
    init_db,
)


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self) -> None:
        """Create a temporary directory and test database for isolated testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_rag_bench.db"
        init_db(self.test_db_path)

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def test_schema_initialization_creates_all_tables(self) -> None:
        """Verify that all four required tables are created by schema.sql."""
        with get_db_connection(self.test_db_path) as conn:
            tables = get_table_names(conn)

        expected_tables = {
            "documents",
            "document_chunks",
            "benchmark_runs",
            "run_evaluations",
        }
        self.assertTrue(
            expected_tables.issubset(set(tables)),
            f"Expected tables {expected_tables} to be present, found {tables}",
        )

    def test_foreign_key_enforcement_on_chunks(self) -> None:
        """Verify that inserting a chunk with an invalid document_id raises IntegrityError."""
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            # Attempting to insert a chunk with a non-existent document_id
            with self.assertRaises(sqlite3.IntegrityError):
                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, content, token_count
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    ("chk_01", "non_existent_doc", 0, "Test chunk content", 10),
                )

    def test_cascade_delete_on_documents(self) -> None:
        """Verify that deleting a document cascades and removes associated chunks."""
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()

            # Insert parent document
            cursor.execute(
                """
                INSERT INTO documents (id, title, source, content)
                VALUES (?, ?, ?, ?);
                """,
                ("doc_01", "Policy 101", "policy.txt", "Full text of policy"),
            )

            # Insert child chunk
            cursor.execute(
                """
                INSERT INTO document_chunks (
                    id, document_id, chunk_index, content, token_count
                ) VALUES (?, ?, ?, ?, ?);
                """,
                ("chk_01", "doc_01", 0, "Chunk text of policy", 5),
            )

        # Reopen and delete parent document
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?;", ("doc_01",))

        # Assert chunk was deleted by ON DELETE CASCADE
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks WHERE id = ?;", ("chk_01",))
            row = cursor.fetchone()
            self.assertEqual(row["cnt"], 0)

    def test_benchmark_run_and_evaluations_insert(self) -> None:
        """Verify inserting benchmark runs and query evaluation metrics."""
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()

            # Insert benchmark run
            cursor.execute(
                """
                INSERT INTO benchmark_runs (
                    id, run_name, config_name, top_k, total_queries, notes
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("run_001", "Baseline Run", "dense_all_minilm", 3, 1, "Initial test run"),
            )

            # Insert evaluation metric record
            cursor.execute(
                """
                INSERT INTO run_evaluations (
                    run_id, query_id, query_text, retrieved_chunk_ids_json,
                    ground_truth_chunk_ids_json, precision_at_k, recall_at_k,
                    reciprocal_rank, groundedness_score, answer_text,
                    is_abstention, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "run_001",
                    "q_01",
                    "What is the late submission penalty?",
                    '["chk_01", "chk_02"]',
                    '["chk_01"]',
                    0.5,
                    1.0,
                    1.0,
                    0.92,
                    "Late submissions incur a 10% penalty per day.",
                    0,
                    42.5,
                ),
            )

        # Verify query and row factory accessibility
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.run_name, e.query_id, e.precision_at_k, e.groundedness_score
                FROM benchmark_runs r
                JOIN run_evaluations e ON r.id = e.run_id
                WHERE r.id = ?;
                """,
                ("run_001",),
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["run_name"], "Baseline Run")
            self.assertEqual(row["query_id"], "q_01")
            self.assertEqual(row["precision_at_k"], 0.5)
            self.assertEqual(row["groundedness_score"], 0.92)

    def test_database_directory_auto_creation(self) -> None:
        """Verify that get_db_path creates parent directories automatically if missing."""
        nested_db = Path(self.temp_dir.name) / "nested_dir" / "deeper" / "bench.db"
        self.assertFalse(nested_db.parent.exists())

        resolved = get_db_path(nested_db)
        self.assertTrue(resolved.parent.exists())


if __name__ == "__main__":
    unittest.main()
