"""
Tests for the CLI interface (DAI-009 Phase 1).

Tests:
1. Asking an answerable question via CLI flags.
2. JSON output format formatting and structure.
3. Dense retrieval execution via CLI flag.
4. Abstention reporting on out-of-domain queries.
5. Parameter validation and missing question handling.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from db.database import init_db
from src.cli import build_parser, format_text_output, main
from src.ingest import ingest_documents
from src.service import RAGResponse, RAGService


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "cli_test.db"
        init_db(self.test_db_path)
        ingest_documents(db_path=self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parser_defaults(self) -> None:
        """Verify CLI parser has correct default flags."""
        parser = build_parser()
        args = parser.parse_args(["-q", "Sample query"])
        self.assertEqual(args.question, "Sample query")
        self.assertEqual(args.retriever, "lexical")
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.format, "text")

    def test_cli_answerable_query_text_format(self) -> None:
        """Verify text output contains answer, status, and retrieved chunks."""
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([
                "-q", "What is the primary WPA3 wireless network SSID for students?",
                "--retriever", "lexical",
                "--db-path", str(self.test_db_path),
            ])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Campus-Secure-5G", output)
        self.assertIn("Abstention Triggered : NO", output)
        self.assertIn("RETRIEVED CONTEXT CHUNKS", output)
        self.assertIn("DOC-IT-001", output)

    def test_cli_unanswerable_query_json_format(self) -> None:
        """Verify json output parses and shows abstention."""
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([
                "-q", "What are the recreational swimming pool lane hours and lap reservation rules?",
                "--format", "json",
                "--db-path", str(self.test_db_path),
            ])

        self.assertEqual(code, 0)
        output = stdout.getvalue().strip()
        data = json.loads(output)
        self.assertTrue(data["is_abstention"])
        self.assertEqual(data["source_chunk_ids"], [])
        self.assertEqual(data["source_doc_ids"], [])
        self.assertIn("cannot find sufficient information", data["answer"].lower())

    def test_cli_top_k_flag(self) -> None:
        """Verify -k flag limits retrieved chunk count."""
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([
                "-q", "Tell me about Wi-Fi",
                "-k", "1",
                "--format", "json",
                "--db-path", str(self.test_db_path),
            ])

        self.assertEqual(code, 0)
        data = json.loads(stdout.getvalue().strip())
        self.assertEqual(len(data["retrieved_chunks"]), 1)

    def test_cli_dense_retrieval(self) -> None:
        """Verify --retriever dense runs dense retrieval."""
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([
                "-q", "computer cluster GPU access",
                "--retriever", "dense",
                "--format", "json",
                "--db-path", str(self.test_db_path),
            ])

        self.assertEqual(code, 0)
        data = json.loads(stdout.getvalue().strip())
        self.assertIn("dense_cosine", data["retriever_name"])

    def test_cli_empty_question_returns_error(self) -> None:
        """Verify empty question yields non-zero exit code."""
        stderr = io.StringIO()
        with patch("sys.stderr", stderr), patch("sys.stdin.isatty", return_value=True):
            code = main(["-q", "   "])

        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
