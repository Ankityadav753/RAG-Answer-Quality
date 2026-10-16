"""
Tests for Render deployment readiness and environment configurations.

Validates:
1. run_app.py host and port resolution (listening on 0.0.0.0 and PORT in cloud/Render environment).
2. Local default host and port preservation (127.0.0.1:8000).
3. Database path configuration via RAG_DB_PATH.
4. Upload directory configuration via RAG_UPLOAD_DIR.
5. Sample documents path configuration via RAG_DOCS_PATH.
6. Offline deterministic mode defaults without requiring external API keys.
7. Health check endpoint (/health) readiness reporting.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from db.database import DEFAULT_DB_PATH, get_db_path
from src.config import DEFAULT_DB_PATH as CONFIG_DB_PATH, DEFAULT_UPLOAD_DIR, get_settings
from src.service import RAGService


class TestDeploymentConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_run_app_defaults_locally(self) -> None:
        """Verify run_app defaults to 127.0.0.1:8000 when PORT is not set."""
        os.environ.pop("PORT", None)
        os.environ.pop("RAG_APP_PORT", None)
        os.environ.pop("RAG_APP_HOST", None)

        from run_app import main

        # Mock uvicorn.run to capture parameters without actually launching the server
        with patch("uvicorn.run") as mock_uvicorn:
            with patch("sys.argv", ["run_app.py"]):
                code = main()
                self.assertEqual(code, 0)
                mock_uvicorn.assert_called_once()
                kwargs = mock_uvicorn.call_args.kwargs
                self.assertEqual(kwargs["host"], "127.0.0.1")
                self.assertEqual(kwargs["port"], 8000)

    def test_run_app_listens_on_render_port_and_all_interfaces(self) -> None:
        """Verify run_app listens on 0.0.0.0 and PORT when PORT environment variable is present."""
        os.environ["PORT"] = "10000"
        os.environ.pop("RAG_APP_PORT", None)
        os.environ.pop("RAG_APP_HOST", None)

        from run_app import main

        with patch("uvicorn.run") as mock_uvicorn:
            with patch("sys.argv", ["run_app.py"]):
                code = main()
                self.assertEqual(code, 0)
                mock_uvicorn.assert_called_once()
                kwargs = mock_uvicorn.call_args.kwargs
                self.assertEqual(kwargs["host"], "0.0.0.0")
                self.assertEqual(kwargs["port"], 10000)

    def test_run_app_respects_explicit_cli_arguments(self) -> None:
        """Verify CLI flags override environment defaults."""
        os.environ["PORT"] = "10000"

        from run_app import main

        with patch("uvicorn.run") as mock_uvicorn:
            with patch("sys.argv", ["run_app.py", "--host", "192.168.1.5", "--port", "9090"]):
                code = main()
                self.assertEqual(code, 0)
                mock_uvicorn.assert_called_once()
                kwargs = mock_uvicorn.call_args.kwargs
                self.assertEqual(kwargs["host"], "192.168.1.5")
                self.assertEqual(kwargs["port"], 9090)

    def test_database_path_configurable_via_env(self) -> None:
        """Verify RAG_DB_PATH environment variable overrides default database path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_db = Path(tmp_dir) / "custom_render.db"
            os.environ["RAG_DB_PATH"] = str(custom_db)

            resolved = get_db_path()
            self.assertEqual(resolved, custom_db.resolve())

            settings = get_settings()
            self.assertEqual(settings.db_path, custom_db.resolve())

    def test_database_path_preserves_local_default(self) -> None:
        """Verify get_db_path falls back to DEFAULT_DB_PATH when RAG_DB_PATH is unset."""
        os.environ.pop("RAG_DB_PATH", None)
        resolved = get_db_path()
        self.assertEqual(resolved, DEFAULT_DB_PATH.resolve())

    def test_upload_dir_configurable_via_env(self) -> None:
        """Verify RAG_UPLOAD_DIR environment variable configures upload storage path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_upload = Path(tmp_dir) / "custom_uploads"
            os.environ["RAG_UPLOAD_DIR"] = str(custom_upload)

            settings = get_settings()
            self.assertEqual(settings.upload_dir, custom_upload.resolve())

            service = RAGService.from_env(auto_init=False)
            self.assertEqual(service.upload_dir, custom_upload.resolve())
            self.assertEqual(service.document_manager.upload_dir, custom_upload.resolve())

    def test_docs_path_configurable_via_env(self) -> None:
        """Verify RAG_DOCS_PATH environment variable configures sample corpus path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_docs = Path(tmp_dir) / "custom_docs.json"
            os.environ["RAG_DOCS_PATH"] = str(custom_docs)

            settings = get_settings()
            self.assertEqual(settings.docs_path, custom_docs.resolve())

            service = RAGService.from_env(auto_init=False)
            self.assertEqual(service.docs_path, custom_docs.resolve())

    def test_offline_mode_requires_no_api_keys(self) -> None:
        """Verify service initializes and queries deterministically without any API keys."""
        os.environ["RAG_MODE"] = "offline"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = Path(tmp_dir) / "offline_test.db"
            service = RAGService(db_path=test_db, auto_init=True)
            self.assertTrue(service.is_mock)
            self.assertGreater(len(service.list_documents()), 0)

            # Query should succeed without network or credentials
            resp = service.query("What is the primary WPA3 wireless network SSID for students?")
            self.assertFalse(resp.is_abstention)
            self.assertIn("Campus-Secure-5G", resp.answer)

    def test_health_check_endpoint(self) -> None:
        """Verify FastAPI /health endpoint returns 200 OK and health response."""
        from fastapi.testclient import TestClient
        from src.app import app

        client = TestClient(app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreater(data["documents_count"], 0)
        self.assertGreater(data["chunks_count"], 0)


if __name__ == "__main__":
    unittest.main()
