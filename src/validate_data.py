"""
Dataset Validation Module for DAI-009: RAG Answer Quality Benchmark.

Validates:
1. JSON formatting and schema compliance.
2. Unique document and question identifiers.
3. Non-empty text fields across all entities.
4. Foreign-key style document reference integrity (relevant_doc_ids exist).
5. Answerability consistency (unanswerable questions have empty doc references).
6. Presence of at least 3 unanswerable / out-of-domain questions for abstention testing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Base paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_PATH: Path = PROJECT_ROOT / "data" / "sample_documents.json"
DEFAULT_QA_PATH: Path = PROJECT_ROOT / "data" / "ground_truth_qa.json"

REQUIRED_DOC_FIELDS = {"id", "title", "category", "source", "content"}
REQUIRED_QA_FIELDS = {
    "id",
    "question",
    "expected_answer",
    "relevant_doc_ids",
    "question_type",
    "is_answerable",
    "key_facts",
}


class DatasetValidationError(Exception):
    """Raised when dataset validation checks fail."""
    def __init__(self, errors: List[str]):
        super().__init__(f"Dataset validation failed with {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors))
        self.errors = errors


def load_json_file(file_path: Path) -> Any:
    """Load and parse a JSON file, returning parsed content or raising informative error."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e


def validate_documents(
    docs_data: Any,
    min_docs: int = 10,
    max_docs: int = 12,
) -> Tuple[Set[str], List[str]]:
    """
    Validate documents list structure and constraints.
    
    Returns:
        Tuple[Set[str], List[str]]: Set of valid document IDs, and list of validation errors.
    """
    errors: List[str] = []
    doc_ids: Set[str] = set()

    if not isinstance(docs_data, list):
        errors.append(f"Documents file must contain a JSON array, got {type(docs_data).__name__}.")
        return doc_ids, errors

    if not (min_docs <= len(docs_data) <= max_docs):
        errors.append(
            f"Expected between {min_docs} and {max_docs} documents, found {len(docs_data)}."
        )

    for idx, doc in enumerate(docs_data):
        if not isinstance(doc, dict):
            errors.append(f"Document at index {idx} is not a JSON object.")
            continue

        missing = REQUIRED_DOC_FIELDS - set(doc.keys())
        if missing:
            errors.append(f"Document at index {idx} missing required fields: {sorted(missing)}.")

        doc_id = doc.get("id")
        if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
            errors.append(f"Document at index {idx} has missing or empty 'id'.")
        elif doc_id in doc_ids:
            errors.append(f"Duplicate document ID detected: '{doc_id}'.")
        else:
            doc_ids.add(doc_id)

        for field in ("title", "category", "source", "content"):
            val = doc.get(field)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"Document '{doc_id or idx}' has missing or empty '{field}'.")

    return doc_ids, errors


def validate_questions(
    qa_data: Any,
    valid_doc_ids: Set[str],
    min_questions: int = 20,
    min_unanswerable: int = 3,
) -> Tuple[Set[str], int, List[str]]:
    """
    Validate benchmark questions structure, references, and answerability.
    
    Returns:
        Tuple[Set[str], int, List[str]]: Set of question IDs, unanswerable count, and errors.
    """
    errors: List[str] = []
    question_ids: Set[str] = set()
    unanswerable_count = 0

    if not isinstance(qa_data, list):
        errors.append(f"QA benchmark file must contain a JSON array, got {type(qa_data).__name__}.")
        return question_ids, unanswerable_count, errors

    if len(qa_data) < min_questions:
        errors.append(f"Expected at least {min_questions} questions, found {len(qa_data)}.")

    for idx, q in enumerate(qa_data):
        if not isinstance(q, dict):
            errors.append(f"Question at index {idx} is not a JSON object.")
            continue

        missing = REQUIRED_QA_FIELDS - set(q.keys())
        if missing:
            errors.append(f"Question at index {idx} missing required fields: {sorted(missing)}.")

        q_id = q.get("id")
        if not q_id or not isinstance(q_id, str) or not q_id.strip():
            errors.append(f"Question at index {idx} has missing or empty 'id'.")
        elif q_id in question_ids:
            errors.append(f"Duplicate question ID detected: '{q_id}'.")
        else:
            question_ids.add(q_id)

        for text_field in ("question", "expected_answer", "question_type"):
            val = q.get(text_field)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"Question '{q_id or idx}' has missing or empty '{text_field}'.")

        key_facts = q.get("key_facts")
        if not isinstance(key_facts, list) or len(key_facts) == 0:
            errors.append(f"Question '{q_id or idx}' must have a non-empty list for 'key_facts'.")

        is_ans = q.get("is_answerable")
        if not isinstance(is_ans, bool):
            errors.append(f"Question '{q_id or idx}' field 'is_answerable' must be a boolean.")
            continue

        rel_docs = q.get("relevant_doc_ids")
        if not isinstance(rel_docs, list):
            errors.append(f"Question '{q_id or idx}' field 'relevant_doc_ids' must be a list.")
            continue

        if not is_ans:
            unanswerable_count += 1
            if len(rel_docs) > 0:
                errors.append(
                    f"Question '{q_id}' is marked unanswerable but specifies relevant_doc_ids: {rel_docs}."
                )
        else:
            if len(rel_docs) == 0:
                errors.append(
                    f"Question '{q_id}' is marked answerable but has empty 'relevant_doc_ids'."
                )
            for r_id in rel_docs:
                if r_id not in valid_doc_ids:
                    errors.append(
                        f"Question '{q_id}' references unknown document ID: '{r_id}'."
                    )

    if unanswerable_count < min_unanswerable:
        errors.append(
            f"Expected at least {min_unanswerable} unanswerable questions, found {unanswerable_count}."
        )

    return question_ids, unanswerable_count, errors


def validate_datasets(
    docs_path: Optional[Path] = None,
    qa_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full end-to-end dataset validation runner.
    
    Returns:
        Dict[str, Any]: Validation summary dictionary.
    """
    target_docs_path = docs_path or DEFAULT_DOCS_PATH
    target_qa_path = qa_path or DEFAULT_QA_PATH

    errors: List[str] = []

    try:
        docs_data = load_json_file(target_docs_path)
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to load documents: {e}"],
            "document_count": 0,
            "question_count": 0,
            "unanswerable_count": 0,
        }

    try:
        qa_data = load_json_file(target_qa_path)
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to load QA dataset: {e}"],
            "document_count": len(docs_data) if isinstance(docs_data, list) else 0,
            "question_count": 0,
            "unanswerable_count": 0,
        }

    doc_ids, doc_errors = validate_documents(docs_data)
    errors.extend(doc_errors)

    q_ids, unanswerable_cnt, qa_errors = validate_questions(qa_data, doc_ids)
    errors.extend(qa_errors)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "document_count": len(doc_ids),
        "question_count": len(q_ids),
        "unanswerable_count": unanswerable_cnt,
        "docs_path": str(target_docs_path),
        "qa_path": str(target_qa_path),
    }


def main() -> int:
    """CLI entrypoint for dataset validation."""
    print("[INFO] Validating RAG benchmark datasets...")
    result = validate_datasets()

    print(f" - Documents Path : {result.get('docs_path')}")
    print(f" - QA Dataset Path: {result.get('qa_path')}")
    print(f" - Document Count : {result.get('document_count')}")
    print(f" - Question Count : {result.get('question_count')}")
    print(f" - Unanswerable   : {result.get('unanswerable_count')}")

    if result["valid"]:
        print("\n[SUCCESS] All dataset validation checks passed cleanly!")
        return 0
    else:
        print(f"\n[FAILURE] Found {len(result['errors'])} validation error(s):")
        for err in result["errors"]:
            print(f"  * {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
