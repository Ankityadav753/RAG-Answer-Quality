"""
Unit and integration tests for dataset integrity and validation rules (DAI-009).

Tests:
1. Real benchmark dataset files pass all validation constraints.
2. Validator properly catches unknown document references.
3. Validator catches duplicate IDs (documents and questions).
4. Validator catches missing required fields.
5. Validator catches answerability mismatches and insufficient unanswerable questions.
6. Validator catches invalid/corrupt JSON files.

Compatible with both `python -m unittest` and `pytest`.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.validate_data import (
    DEFAULT_DOCS_PATH,
    DEFAULT_QA_PATH,
    load_json_file,
    validate_datasets,
    validate_documents,
    validate_questions,
)


class TestDatasetIntegrity(unittest.TestCase):
    def test_real_benchmark_datasets_pass_validation(self) -> None:
        """Verify that the actual data/sample_documents.json and data/ground_truth_qa.json pass."""
        result = validate_datasets(DEFAULT_DOCS_PATH, DEFAULT_QA_PATH)
        self.assertTrue(
            result["valid"],
            f"Expected dataset to be valid, but got errors: {result.get('errors')}",
        )
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(result["document_count"], 11)
        self.assertEqual(result["question_count"], 22)
        self.assertEqual(result["unanswerable_count"], 3)

    def test_detects_unknown_document_reference(self) -> None:
        """Verify that questions referencing non-existent documents are flagged."""
        valid_doc_ids = {"DOC-IT-001"}
        bad_qa = [
            {
                "id": "Q-TEST-01",
                "question": "What is the Wi-Fi password?",
                "expected_answer": "No password.",
                "relevant_doc_ids": ["DOC-IT-001", "DOC-GHOST-999"],
                "question_type": "keyword_identifier",
                "is_answerable": True,
                "key_facts": ["No password"],
            }
        ]
        _, _, errors = validate_questions(
            bad_qa, valid_doc_ids, min_questions=1, min_unanswerable=0
        )
        self.assertTrue(
            any("DOC-GHOST-999" in err for err in errors),
            f"Expected unknown document error, got: {errors}",
        )

    def test_detects_duplicate_document_ids(self) -> None:
        """Verify duplicate document IDs trigger validation errors."""
        duplicate_docs = [
            {
                "id": "DOC-DUP",
                "title": "Doc One",
                "category": "Cat A",
                "source": "s1.md",
                "content": "Content one",
            },
            {
                "id": "DOC-DUP",
                "title": "Doc Two",
                "category": "Cat B",
                "source": "s2.md",
                "content": "Content two",
            },
        ]
        _, errors = validate_documents(duplicate_docs, min_docs=2, max_docs=2)
        self.assertTrue(
            any("Duplicate document ID" in err for err in errors),
            f"Expected duplicate error, got: {errors}",
        )

    def test_detects_duplicate_question_ids(self) -> None:
        """Verify duplicate question IDs trigger validation errors."""
        duplicate_qa = [
            {
                "id": "Q-DUP",
                "question": "Question 1?",
                "expected_answer": "Answer 1",
                "relevant_doc_ids": ["DOC-01"],
                "question_type": "keyword",
                "is_answerable": True,
                "key_facts": ["Fact 1"],
            },
            {
                "id": "Q-DUP",
                "question": "Question 2?",
                "expected_answer": "Answer 2",
                "relevant_doc_ids": ["DOC-01"],
                "question_type": "keyword",
                "is_answerable": True,
                "key_facts": ["Fact 2"],
            },
        ]
        _, _, errors = validate_questions(
            duplicate_qa, {"DOC-01"}, min_questions=2, min_unanswerable=0
        )
        self.assertTrue(
            any("Duplicate question ID" in err for err in errors),
            f"Expected duplicate question error, got: {errors}",
        )

    def test_detects_missing_required_fields(self) -> None:
        """Verify that missing document fields (e.g. content) are flagged."""
        incomplete_docs = [
            {
                "id": "DOC-INCOMPLETE",
                "title": "Title only",
                # missing category, source, content
            }
        ]
        _, errors = validate_documents(incomplete_docs, min_docs=1, max_docs=1)
        self.assertTrue(
            any("missing required fields" in err for err in errors),
            f"Expected missing fields error, got: {errors}",
        )

    def test_detects_answerability_mismatch(self) -> None:
        """Verify unanswerable question with doc IDs or answerable question with empty doc IDs are flagged."""
        # Case A: unanswerable with doc IDs
        qa_with_illegal_doc_ids = [
            {
                "id": "Q-UNANS-FAIL",
                "question": "Can I park on the roof?",
                "expected_answer": "Abstain",
                "relevant_doc_ids": ["DOC-01"],
                "question_type": "unanswerable_abstention",
                "is_answerable": False,
                "key_facts": ["Out of domain"],
            }
        ]
        _, _, errors_a = validate_questions(
            qa_with_illegal_doc_ids, {"DOC-01"}, min_questions=1, min_unanswerable=1
        )
        self.assertTrue(
            any("marked unanswerable but specifies relevant_doc_ids" in err for err in errors_a),
            f"Expected unanswerable mismatch error, got: {errors_a}",
        )

        # Case B: answerable with empty doc IDs
        qa_with_empty_doc_ids = [
            {
                "id": "Q-ANS-FAIL",
                "question": "What is the Wi-Fi SSID?",
                "expected_answer": "Campus-Secure-5G",
                "relevant_doc_ids": [],
                "question_type": "keyword",
                "is_answerable": True,
                "key_facts": ["SSID"],
            }
        ]
        _, _, errors_b = validate_questions(
            qa_with_empty_doc_ids, {"DOC-01"}, min_questions=1, min_unanswerable=0
        )
        self.assertTrue(
            any("marked answerable but has empty 'relevant_doc_ids'" in err for err in errors_b),
            f"Expected empty relevant_doc_ids error, got: {errors_b}",
        )

    def test_detects_insufficient_unanswerable_questions(self) -> None:
        """Verify error is raised when unanswerable count is below minimum threshold."""
        qa_only_answerable = [
            {
                "id": f"Q-{i}",
                "question": f"Question {i}?",
                "expected_answer": "Answer",
                "relevant_doc_ids": ["DOC-01"],
                "question_type": "keyword",
                "is_answerable": True,
                "key_facts": ["Fact"],
            }
            for i in range(20)
        ]
        _, unans_cnt, errors = validate_questions(
            qa_only_answerable, {"DOC-01"}, min_questions=20, min_unanswerable=3
        )
        self.assertEqual(unans_cnt, 0)
        self.assertTrue(
            any("Expected at least 3 unanswerable questions" in err for err in errors),
            f"Expected insufficient unanswerable error, got: {errors}",
        )

    def test_detects_malformed_json_file(self) -> None:
        """Verify validator handles non-JSON / corrupt files cleanly."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            f.write("{corrupt json content...")
            corrupt_path = Path(f.name)

        try:
            result = validate_datasets(docs_path=corrupt_path)
            self.assertFalse(result["valid"])
            self.assertTrue(any("Failed to load documents" in err for err in result["errors"]))
        finally:
            corrupt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
