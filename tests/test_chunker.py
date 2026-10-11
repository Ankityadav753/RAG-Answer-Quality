"""
Unit tests for deterministic text chunking (DAI-009).

Tests:
1. Chunk boundaries and segmentation logic.
2. Word-level overlap between consecutive chunks.
3. Empty, whitespace-only, and non-string inputs.
4. Input smaller than chunk size.
5. Parameter validation (invalid chunk size, negative overlap, overlap >= size).

Compatible with both `python -m unittest` and `pytest`.
"""

import unittest

from src.chunker import TextChunk, chunk_text, validate_chunking_params


class TestTextChunker(unittest.TestCase):
    def test_chunk_boundaries(self) -> None:
        """Verify that words are properly segmented across chunk boundaries without truncation."""
        words = [f"word_{i}" for i in range(25)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        # step = 8
        # Chunk 0: indices 0..10 (10 words)
        # Chunk 1: indices 8..18 (10 words)
        # Chunk 2: indices 16..25 (9 words)
        self.assertEqual(len(chunks), 3)

        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].token_count, 10)
        self.assertEqual(chunks[0].content.split(), words[0:10])

        self.assertEqual(chunks[1].chunk_index, 1)
        self.assertEqual(chunks[1].token_count, 10)
        self.assertEqual(chunks[1].content.split(), words[8:18])

        self.assertEqual(chunks[2].chunk_index, 2)
        self.assertEqual(chunks[2].token_count, 9)
        self.assertEqual(chunks[2].content.split(), words[16:25])

    def test_chunk_overlap(self) -> None:
        """Verify that adjacent chunks share the exact specified overlapping words."""
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
        # step = 3
        # chunk 0: one two three four five
        # chunk 1: four five six seven eight
        # chunk 2: seven eight nine ten
        self.assertEqual(len(chunks), 3)

        chunk0_words = chunks[0].content.split()
        chunk1_words = chunks[1].content.split()
        chunk2_words = chunks[2].content.split()

        self.assertEqual(chunk0_words[-2:], chunk1_words[:2])
        self.assertEqual(chunk1_words[-2:], chunk2_words[:2])

    def test_empty_and_whitespace_text(self) -> None:
        """Verify that empty or whitespace strings return an empty list of chunks."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])
        self.assertEqual(chunk_text(None), [])  # type: ignore

    def test_text_smaller_than_chunk_size(self) -> None:
        """Verify that text shorter than chunk_size yields exactly one chunk."""
        short_text = "University Wi-Fi policy is simple."
        chunks = chunk_text(short_text, chunk_size=50, chunk_overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].content, short_text)
        self.assertEqual(chunks[0].token_count, 5)

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Verify that invalid chunk sizes or overlap values raise clear exceptions."""
        # chunk_size <= 0
        with self.assertRaises(ValueError):
            chunk_text("valid text", chunk_size=0, chunk_overlap=0)

        with self.assertRaises(ValueError):
            chunk_text("valid text", chunk_size=-10, chunk_overlap=0)

        # chunk_overlap < 0
        with self.assertRaises(ValueError):
            chunk_text("valid text", chunk_size=50, chunk_overlap=-5)

        # chunk_overlap >= chunk_size
        with self.assertRaises(ValueError):
            chunk_text("valid text", chunk_size=20, chunk_overlap=20)

        with self.assertRaises(ValueError):
            chunk_text("valid text", chunk_size=20, chunk_overlap=25)

    def test_parameter_type_validation(self) -> None:
        """Verify that non-integer arguments raise TypeError."""
        with self.assertRaises(TypeError):
            validate_chunking_params("50", 10)  # type: ignore

        with self.assertRaises(TypeError):
            validate_chunking_params(50, "10")  # type: ignore

        with self.assertRaises(TypeError):
            validate_chunking_params(True, 10)  # type: ignore


if __name__ == "__main__":
    unittest.main()
