"""
Deterministic text chunking module for DAI-009: RAG Answer Quality Benchmark.

Provides configurable sliding-window text chunking with word-boundary preservation,
parameter validation, and structured metadata.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TextChunk:
    """Represents a discrete text chunk with position and token metadata."""
    chunk_index: int
    content: str
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary representation."""
        return asdict(self)


def validate_chunking_params(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate that chunking parameters satisfy algorithmic constraints.
    
    Raises:
        TypeError: If parameters are not integers.
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def chunk_text(
    text: str,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
) -> List[TextChunk]:
    """
    Split input text into overlapping chunks using word-level boundaries.
    
    Args:
        text: Raw input text to segment.
        chunk_size: Maximum number of words per chunk (default: 80).
        chunk_overlap: Number of overlapping words between consecutive chunks (default: 20).
        
    Returns:
        List[TextChunk]: Ordered list of text chunks. Returns empty list for empty/whitespace input.
    """
    validate_chunking_params(chunk_size, chunk_overlap)

    if not text or not isinstance(text, str):
        return []

    # Normalize whitespace and tokenize by words
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    words = re.split(r"\s+", cleaned_text)
    total_words = len(words)

    if total_words == 0:
        return []

    # If the text is shorter than or equal to chunk_size, return a single chunk
    if total_words <= chunk_size:
        return [
            TextChunk(
                chunk_index=0,
                content=" ".join(words),
                token_count=total_words,
            )
        ]

    step = chunk_size - chunk_overlap
    chunks: List[TextChunk] = []
    chunk_idx = 0
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append(
            TextChunk(
                chunk_index=chunk_idx,
                content=chunk_str,
                token_count=len(chunk_words),
            )
        )
        chunk_idx += 1

        # If this chunk reached the end of the text, stop
        if end >= total_words:
            break

        # Advance start index by step
        start += step

    return chunks
