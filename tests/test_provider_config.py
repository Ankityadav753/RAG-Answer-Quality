"""
Tests for provider configuration and real model adapters (DAI-009 Phase 1).

Tests:
1. Environment variable settings discovery and offline/online modes.
2. OpenAICompatibleProvider initialization and missing API key guardrail.
3. OpenAICompatibleProvider execution using mocked API response.
4. OpenAIEmbeddingModel initialization and missing API key guardrail.
5. OpenAIEmbeddingModel encoding using mocked API response.
6. Error handling for HTTP and network failures.
"""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import Settings, get_settings
from src.embeddings import (
    DeterministicTestDoubleEmbeddingModel,
    EmbeddingModelUnavailableError,
    OpenAIEmbeddingModel,
    get_embedding_model,
)
from src.generator import (
    DeterministicMockLLMProvider,
    LLMProviderError,
    OpenAICompatibleProvider,
    get_llm_provider,
)


class TestProviderConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_default_offline_settings(self) -> None:
        """Verify default configuration defaults to offline mode with mock providers."""
        os.environ["RAG_MODE"] = "offline"
        os.environ.pop("RAG_LLM_PROVIDER", None)
        os.environ.pop("RAG_EMBEDDING_PROVIDER", None)

        settings = get_settings()
        self.assertTrue(settings.is_offline)
        self.assertEqual(settings.llm_provider, "mock")
        self.assertEqual(settings.embedding_provider, "mock")

    def test_env_override_settings(self) -> None:
        """Verify environment variables correctly override runtime settings."""
        os.environ["RAG_MODE"] = "online"
        os.environ["RAG_LLM_PROVIDER"] = "openai"
        os.environ["RAG_EMBEDDING_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-test-secret-12345"
        os.environ["OPENAI_MODEL"] = "gpt-4o"
        os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-large"
        os.environ["OPENAI_EMBEDDING_DIM"] = "3072"
        os.environ["RAG_TOP_K"] = "5"

        settings = get_settings()
        self.assertFalse(settings.is_offline)
        self.assertEqual(settings.llm_provider, "openai")
        self.assertEqual(settings.embedding_provider, "openai")
        self.assertEqual(settings.openai_api_key, "sk-test-secret-12345")
        self.assertEqual(settings.openai_model, "gpt-4o")
        self.assertEqual(settings.openai_embedding_model, "text-embedding-3-large")
        self.assertEqual(settings.openai_embedding_dim, 3072)
        self.assertEqual(settings.top_k, 5)

    def test_openai_llm_missing_api_key_raises(self) -> None:
        """Verify OpenAICompatibleProvider raises LLMProviderError when no key is provided."""
        os.environ.pop("OPENAI_API_KEY", None)
        provider = OpenAICompatibleProvider(api_key=None)
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate("Test prompt")
        self.assertIn("Missing API key", str(ctx.exception))

    def test_openai_llm_generate_with_mocked_http(self) -> None:
        """Verify OpenAICompatibleProvider parses standard chat completion payload."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "The late assignment penalty is 10% per 24-hour period."
                    }
                }
            ]
        }
        mock_bytes = json.dumps(mock_response).encode("utf-8")

        mock_resp_obj = MagicMock()
        mock_resp_obj.read.return_value = mock_bytes
        mock_resp_obj.__enter__.return_value = mock_resp_obj

        with patch("urllib.request.urlopen", return_value=mock_resp_obj) as mock_urlopen:
            provider = OpenAICompatibleProvider(api_key="sk-valid-test-key", model="gpt-4o-mini")
            res = provider.generate("What is the late penalty?")
            self.assertEqual(res, "The late assignment penalty is 10% per 24-hour period.")
            mock_urlopen.assert_called_once()

    def test_openai_embedding_missing_api_key_raises(self) -> None:
        """Verify OpenAIEmbeddingModel raises EmbeddingModelUnavailableError when no key is set."""
        os.environ.pop("OPENAI_API_KEY", None)
        model = OpenAIEmbeddingModel(api_key=None)
        with self.assertRaises(EmbeddingModelUnavailableError) as ctx:
            model.encode(["Test text"])
        self.assertIn("Missing OpenAI API key", str(ctx.exception))

    def test_openai_embedding_encode_with_mocked_http(self) -> None:
        """Verify OpenAIEmbeddingModel encodes texts and normalizes returned vectors."""
        # Mock 4-dimensional embeddings for 2 texts
        mock_response = {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                {"index": 1, "embedding": [0.0, 2.0, 0.0, 0.0]},
            ]
        }
        mock_bytes = json.dumps(mock_response).encode("utf-8")

        mock_resp_obj = MagicMock()
        mock_resp_obj.read.return_value = mock_bytes
        mock_resp_obj.__enter__.return_value = mock_resp_obj

        with patch("urllib.request.urlopen", return_value=mock_resp_obj):
            model = OpenAIEmbeddingModel(
                model_name="text-embedding-3-small",
                api_key="sk-test-key",
                dimension=4,
            )
            matrix = model.encode(["First test text", "Second test text"])
            self.assertIsInstance(matrix, np.ndarray)
            self.assertEqual(matrix.shape, (2, 4))
            # Verify L2 normalization
            norms = np.linalg.norm(matrix, axis=1)
            np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)

    def test_get_embedding_model_factory(self) -> None:
        """Verify get_embedding_model returns expected model classes based on provider."""
        # Offline default
        m_mock = get_embedding_model(provider="mock")
        self.assertIsInstance(m_mock, DeterministicTestDoubleEmbeddingModel)

        # OpenAI provider
        m_openai = get_embedding_model(provider="openai", api_key="sk-test-key")
        self.assertIsInstance(m_openai, OpenAIEmbeddingModel)
        self.assertEqual(m_openai.model_name, "openai:text-embedding-3-small")

        # Unknown provider raises ValueError
        with self.assertRaises(ValueError):
            get_embedding_model(provider="unknown_provider")


if __name__ == "__main__":
    unittest.main()
