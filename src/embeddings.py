"""
Embedding models module for DAI-009: RAG Answer Quality Benchmark.

Provides:
1. BaseEmbeddingModel abstract interface.
2. SentenceTransformerEmbeddingModel for real transformer-based dense embeddings.
3. DeterministicTestDoubleEmbeddingModel for reproducible, offline, network-free testing.
4. EmbeddingModelUnavailableError for clear setup error reporting when dependencies are missing.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the requested embedding model or its runtime dependencies are unavailable."""
    pass


class BaseEmbeddingModel(ABC):
    """Abstract base class for dense text embedding models."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense embedding vectors for a list of input texts.
        
        Args:
            texts: List of strings to encode.
            
        Returns:
            np.ndarray: 2D array of shape (N, dimension), L2-normalized.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of the embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the model."""
        pass


class DeterministicTestDoubleEmbeddingModel(BaseEmbeddingModel):
    """
    Offline, deterministic embedding model for unit testing and network-free benchmarks.
    
    Generates reproducible 64-dimensional dense vectors using deterministic n-gram
    hashing and projection. Guarantees cosine similarity is higher for texts sharing
    semantic tokens without downloading any external weights.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension
        self._name = "deterministic-test-double-v1"
        # Seeded pseudo-random projection basis for dimension reduction
        rng = np.random.RandomState(42)
        # 1024 hash buckets projected into dimension d
        self._projection_matrix = rng.randn(1024, self._dim).astype(np.float32)
        # Normalize projection columns
        norms = np.linalg.norm(self._projection_matrix, axis=1, keepdims=True)
        self._projection_matrix /= np.maximum(norms, 1e-12)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def _hash_token(self, token: str) -> int:
        """Hash a token to a bucket index [0, 1023]."""
        digest = hashlib.md5(token.lower().encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 1024

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate normalized dense vectors deterministically."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            # Tokenize words and subwords
            tokens = re.findall(r"\b\w+\b", text.lower())
            if not tokens:
                continue

            # Bag of hashed features
            bag = np.zeros(1024, dtype=np.float32)
            for token in tokens:
                idx = self._hash_token(token)
                bag[idx] += 1.0
                # Also include character bigrams for subword robustness
                for j in range(len(token) - 2):
                    sub = token[j : j + 3]
                    sub_idx = self._hash_token(sub)
                    bag[sub_idx] += 0.5

            # Project into embedding dimension
            vec = np.dot(bag, self._projection_matrix)

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vectors[i] = vec / norm
            else:
                vectors[i] = vec

        return vectors


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Production dense vector embedding model powered by sentence-transformers.
    
    Requires `sentence-transformers` library and network connectivity on first load.
    Raises EmbeddingModelUnavailableError if dependencies are missing or initialization fails.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dim = 384  # Default dimension for all-MiniLM-L6-v2

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise EmbeddingModelUnavailableError(
                f"sentence-transformers library is not installed: {e}.\n"
                "To enable dense vector retrieval, install: pip install sentence-transformers\n"
                "Or configure use_test_double=True to use the deterministic test double."
            ) from e
        except Exception as e:
            raise EmbeddingModelUnavailableError(
                f"Failed to load sentence-transformer model '{model_name}': {e}.\n"
                "Check network connectivity or use the deterministic test double."
            ) from e

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        if self._model is None:
            raise EmbeddingModelUnavailableError("SentenceTransformer model is not loaded.")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


def get_embedding_model(
    model_name: Optional[str] = None,
    use_test_double: bool = False,
) -> BaseEmbeddingModel:
    """
    Factory helper to instantiate an embedding model.
    
    Args:
        model_name: Name of sentence-transformer model (default: all-MiniLM-L6-v2).
        use_test_double: If True, uses the offline deterministic test double.
        
    Returns:
        BaseEmbeddingModel: Configured model instance.
    """
    if use_test_double or model_name == "deterministic_test_double":
        return DeterministicTestDoubleEmbeddingModel()

    target_name = model_name or "all-MiniLM-L6-v2"
    return SentenceTransformerEmbeddingModel(model_name=target_name)
