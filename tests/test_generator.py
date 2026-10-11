"""
Unit tests for context-grounded answer generation and abstention guardrails (DAI-009).

Tests:
1. Grounded answer generation using facts from retrieved chunks.
2. Missing context (empty retrieval list) triggers automatic abstention.
3. Unanswerable/irrelevant context triggers evidence-based abstention.
4. Source attribution tracks and deduplicates document and chunk IDs.
5. Provider error propagation via LLMProviderError.
6. Empty question inputs trigger safe refusal.
7. Prompt builder formatting and source citation headers.
8. GeneratedAnswer serialization to dictionary.

Compatible with both `python -m unittest` and `pytest`.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.generator import (
    ABSTENTION_MESSAGE,
    AnswerGenerator,
    BaseLLMProvider,
    DeterministicMockLLMProvider,
    GeneratedAnswer,
    LLMProviderError,
    OpenAICompatibleProvider,
    build_prompt,
)
from src.retriever import RetrievalResult


class FailingTestProvider(BaseLLMProvider):
    """Test double provider that deliberately fails."""
    @property
    def provider_name(self) -> str:
        return "failing-test-provider"

    def generate(self, prompt: str) -> str:
        raise LLMProviderError("Simulated LLM network timeout or connection refused")


class TestAnswerGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = AnswerGenerator()

    def test_grounded_answer_generation(self) -> None:
        """Verify generator extracts grounded facts and cites source chunks."""
        chunk = RetrievalResult(
            chunk_id="DOC-IT-001_chk_00",
            document_id="DOC-IT-001",
            content="Students and faculty must connect to the primary SSID named 'Campus-Secure-5G', which utilizes WPA3-Enterprise.",
            score=0.92,
            rank=1,
        )

        res = self.generator.generate(
            query="What is the student Wi-Fi SSID and security type?",
            retrieved_chunks=[chunk],
        )

        self.assertIsInstance(res, GeneratedAnswer)
        self.assertFalse(res.is_abstention)
        self.assertIn("Campus-Secure-5G", res.answer)
        self.assertIn("WPA3-Enterprise", res.answer)
        self.assertEqual(res.source_chunk_ids, ["DOC-IT-001_chk_00"])
        self.assertEqual(res.source_doc_ids, ["DOC-IT-001"])
        self.assertEqual(res.provider, "mock-deterministic-test-double")

    def test_missing_context_triggers_abstention(self) -> None:
        """Verify that passing an empty list of retrieved chunks triggers abstention."""
        res = self.generator.generate(
            query="What is the Wi-Fi password?",
            retrieved_chunks=[],
        )

        self.assertTrue(res.is_abstention)
        self.assertEqual(res.answer, ABSTENTION_MESSAGE)
        self.assertEqual(res.source_chunk_ids, [])
        self.assertEqual(res.source_doc_ids, [])

    def test_unanswerable_query_triggers_abstention(self) -> None:
        """Verify that out-of-domain queries without supporting evidence trigger abstention."""
        # Unrelated chunk about parking permits
        irrelevant_chunk = RetrievalResult(
            chunk_id="DOC-FAC-002_chk_00",
            document_id="DOC-FAC-002",
            content="Zone A Resident parking permits cost $280 per semester for motor vehicles.",
            score=0.15,
            rank=1,
        )

        res = self.generator.generate(
            query="What is the per-diem meal reimbursement rate for academic conferences?",
            retrieved_chunks=[irrelevant_chunk],
        )

        self.assertTrue(res.is_abstention)
        self.assertIn("cannot find sufficient information", res.answer.lower())
        self.assertEqual(res.source_chunk_ids, [])
        self.assertEqual(res.source_doc_ids, [])

    def test_source_attribution_and_deduplication(self) -> None:
        """Verify source chunk and document IDs are preserved and deduplicated."""
        chunks = [
            RetrievalResult(
                chunk_id="DOC-IT-001_chk_00",
                document_id="DOC-IT-001",
                content="SSID is Campus-Secure-5G.",
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                chunk_id="DOC-IT-001_chk_01",
                document_id="DOC-IT-001",
                content="Bandwidth on guest Wi-Fi is capped at 5 Mbps.",
                score=0.8,
                rank=2,
            ),
            RetrievalResult(
                chunk_id="DOC-IT-002_chk_00",
                document_id="DOC-IT-002",
                content="Storage quota is 50 GB.",
                score=0.7,
                rank=3,
            ),
        ]

        res = self.generator.generate(
            query="Tell me about the Wi-Fi SSID and guest bandwidth limit.",
            retrieved_chunks=chunks,
        )

        self.assertFalse(res.is_abstention)
        self.assertEqual(
            res.source_chunk_ids,
            ["DOC-IT-001_chk_00", "DOC-IT-001_chk_01", "DOC-IT-002_chk_00"],
        )
        # Verify document IDs are deduplicated in order of appearance
        self.assertEqual(res.source_doc_ids, ["DOC-IT-001", "DOC-IT-002"])

    def test_empty_query_triggers_immediate_abstention(self) -> None:
        """Verify empty or whitespace-only questions safely abstain without provider calls."""
        res1 = self.generator.generate(query="", retrieved_chunks=[])
        self.assertTrue(res1.is_abstention)
        self.assertEqual(res1.source_chunk_ids, [])

        res2 = self.generator.generate(query="   \t  ", retrieved_chunks=[])
        self.assertTrue(res2.is_abstention)
        self.assertEqual(res2.source_chunk_ids, [])

    def test_provider_error_propagation(self) -> None:
        """Verify that provider failures raise LLMProviderError."""
        failing_gen = AnswerGenerator(provider=FailingTestProvider())
        chunk = RetrievalResult(
            chunk_id="chk_01",
            document_id="doc_01",
            content="Sample text",
            score=0.5,
            rank=1,
        )

        with self.assertRaises(LLMProviderError) as ctx:
            failing_gen.generate(query="Sample question?", retrieved_chunks=[chunk])
        self.assertIn("Simulated LLM network timeout", str(ctx.exception))

    def test_build_prompt_structure(self) -> None:
        """Verify build_prompt generates clear instructional boundaries and citations."""
        chunk = RetrievalResult(
            chunk_id="DOC-ACAD-002_chk_00",
            document_id="DOC-ACAD-002",
            content="Assignments submitted late are penalized 10% per 24 hours.",
            score=0.88,
            rank=1,
        )

        prompt = build_prompt(query="What is the late penalty?", retrieved_chunks=[chunk])

        self.assertIn("--- CONTEXT CHUNKS ---", prompt)
        self.assertIn("[Context 1 | Document: DOC-ACAD-002 | Chunk: DOC-ACAD-002_chk_00]", prompt)
        self.assertIn("Assignments submitted late are penalized 10% per 24 hours.", prompt)
        self.assertIn("--- QUESTION ---\nWhat is the late penalty?", prompt)
        self.assertIn("--- ANSWER ---", prompt)

    def test_generated_answer_to_dict(self) -> None:
        """Verify GeneratedAnswer converts cleanly to dictionary."""
        ans = GeneratedAnswer(
            answer="Test answer text.",
            is_abstention=False,
            source_chunk_ids=["chk_01"],
            source_doc_ids=["doc_01"],
            prompt_used="Prompt text",
            provider="test-provider",
        )
        d = ans.to_dict()
        self.assertEqual(d["answer"], "Test answer text.")
        self.assertFalse(d["is_abstention"])
        self.assertEqual(d["source_chunk_ids"], ["chk_01"])
        self.assertEqual(d["source_doc_ids"], ["doc_01"])
        self.assertEqual(d["provider"], "test-provider")


if __name__ == "__main__":
    unittest.main()
