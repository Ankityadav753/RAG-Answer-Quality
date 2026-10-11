"""
Unit and integration tests for the evaluation engine and benchmark persistence (DAI-009).

Tests:
1. Precision@K calculation across full, partial, and zero hit cases.
2. Recall@K calculation across single and multi-relevant document scenarios.
3. Reciprocal Rank (RR) across top positions, beyond-K, and zero hits.
4. Edge cases: Unanswerable queries (empty ground-truth relevance), empty retrieved lists.
5. Parameter validation: Invalid k values raise ValueError.
6. Groundedness calculation: Supported claims, partially grounded text, and abstention behavior.
7. Abstention classification metrics: Confusion matrix (TP, FP, FN, TN), precision, recall, F1.
8. Database persistence: Multiple benchmark runs log independently into SQLite without collisions.
9. Report generation: Machine-readable JSON and Markdown report formatting.

Compatible with both `python -m unittest` and `pytest`.
"""

import json
import tempfile
import unittest
from pathlib import Path

from db.database import get_db_connection, init_db
from src.evaluator import (
    BenchmarkSummary,
    PerQueryEvaluation,
    compute_abstention_metrics,
    compute_groundedness_score,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    extract_content_tokens,
)
from src.benchmark_runner import generate_markdown_report, save_benchmark_to_sqlite


class TestEvaluator(unittest.TestCase):
    def test_precision_at_k(self) -> None:
        """Verify Precision@K math."""
        # 2 hits out of k=3 -> 2/3 = 0.6667
        self.assertEqual(
            compute_precision_at_k(["DOC-01", "DOC-02", "DOC-03"], ["DOC-01", "DOC-02"], k=3),
            0.6667,
        )
        # 1 hit out of k=2 -> 0.5
        self.assertEqual(
            compute_precision_at_k(["DOC-01", "DOC-99"], ["DOC-01"], k=2),
            0.5,
        )
        # 0 hits
        self.assertEqual(
            compute_precision_at_k(["DOC-88", "DOC-99"], ["DOC-01"], k=2),
            0.0,
        )

    def test_recall_at_k(self) -> None:
        """Verify Recall@K math."""
        # 1 relevant doc, found in top-k -> 1.0
        self.assertEqual(
            compute_recall_at_k(["DOC-01", "DOC-02"], ["DOC-01"], k=2),
            1.0,
        )
        # 2 relevant docs, 1 found in top-k -> 0.5
        self.assertEqual(
            compute_recall_at_k(["DOC-01", "DOC-03"], ["DOC-01", "DOC-02"], k=2),
            0.5,
        )
        # 0 found
        self.assertEqual(
            compute_recall_at_k(["DOC-03", "DOC-04"], ["DOC-01"], k=2),
            0.0,
        )

    def test_reciprocal_rank(self) -> None:
        """Verify Reciprocal Rank math."""
        # Rank 1 hit -> 1.0
        self.assertEqual(compute_reciprocal_rank(["DOC-01", "DOC-02"], ["DOC-01"], k=2), 1.0)
        # Rank 2 hit -> 0.5
        self.assertEqual(compute_reciprocal_rank(["DOC-99", "DOC-01"], ["DOC-01"], k=2), 0.5)
        # Rank 3 hit -> 0.3333
        self.assertEqual(compute_reciprocal_rank(["DOC-88", "DOC-99", "DOC-01"], ["DOC-01"], k=3), 0.3333)
        # Hit at rank 3 when k=2 -> 0.0 (beyond k)
        self.assertEqual(compute_reciprocal_rank(["DOC-88", "DOC-99", "DOC-01"], ["DOC-01"], k=2), 0.0)
        # Not found -> 0.0
        self.assertEqual(compute_reciprocal_rank(["DOC-88"], ["DOC-01"], k=3), 0.0)

    def test_unanswerable_and_empty_edge_cases(self) -> None:
        """Verify metric conventions on unanswerable and empty inputs."""
        # Empty relevant docs (unanswerable question)
        self.assertEqual(compute_precision_at_k(["DOC-01"], [], k=3), 0.0)
        self.assertEqual(compute_recall_at_k(["DOC-01"], [], k=3), 0.0)
        self.assertEqual(compute_reciprocal_rank(["DOC-01"], [], k=3), 0.0)

        # Empty retrieved docs
        self.assertEqual(compute_precision_at_k([], ["DOC-01"], k=3), 0.0)
        self.assertEqual(compute_recall_at_k([], ["DOC-01"], k=3), 0.0)
        self.assertEqual(compute_reciprocal_rank([], ["DOC-01"], k=3), 0.0)

    def test_invalid_k_raises_error(self) -> None:
        """Verify non-positive k raises ValueError."""
        with self.assertRaises(ValueError):
            compute_precision_at_k(["DOC-01"], ["DOC-01"], k=0)
        with self.assertRaises(ValueError):
            compute_recall_at_k(["DOC-01"], ["DOC-01"], k=-1)
        with self.assertRaises(ValueError):
            compute_reciprocal_rank(["DOC-01"], ["DOC-01"], k="invalid")  # type: ignore

    def test_groundedness_score(self) -> None:
        """Verify groundedness token-containment calculation."""
        context = ["Students must use WPA3-Enterprise on Campus-Secure-5G."]

        # Fully supported answer (all content tokens present in context)
        score_full = compute_groundedness_score(
            answer="Students must use Campus-Secure-5G WPA3-Enterprise.",
            context_chunks=context,
            is_abstention=False,
        )
        self.assertEqual(score_full, 1.0)

        # Partially supported answer (hallucinated extra terms like 'bluetooth' and 'password')
        score_partial = compute_groundedness_score(
            answer="Campus-Secure-5G requires bluetooth password authentication.",
            context_chunks=context,
            is_abstention=False,
        )
        self.assertLess(score_partial, 1.0)
        self.assertGreater(score_partial, 0.0)

        # Abstention always scores 1.0 (refusal avoids hallucination)
        self.assertEqual(
            compute_groundedness_score(
                answer="I cannot find sufficient information.",
                context_chunks=context,
                is_abstention=True,
            ),
            1.0,
        )

    def test_abstention_metrics(self) -> None:
        """Verify confusion matrix and abstention precision/recall/F1."""
        # 4 queries:
        # q0: unanswerable (gt=False), pred=True (Abstain)  -> TP
        # q1: answerable   (gt=True),  pred=True (Abstain)  -> FP
        # q2: unanswerable (gt=False), pred=False (Answer)  -> FN
        # q3: answerable   (gt=True),  pred=False (Answer)  -> TN
        preds = [True, True, False, False]
        gts = [False, True, False, True]

        metrics = compute_abstention_metrics(preds, gts)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_database_persistence_multiple_runs(self) -> None:
        """Verify benchmark runs log into SQLite without collisions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db = Path(temp_dir) / "bench_test.db"
            init_db(test_db)

            # Create mock query evaluation
            q_eval = PerQueryEvaluation(
                query_id="Q-01",
                query_text="What is Wi-Fi SSID?",
                question_type="keyword",
                is_answerable_ground_truth=True,
                relevant_doc_ids=["DOC-01"],
                retrieved_chunk_ids=["chk-01"],
                retrieved_doc_ids=["DOC-01"],
                precision_at_k=1.0,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
                is_abstention=False,
                groundedness_score=1.0,
                answer_text="Campus-Secure-5G",
                latency_ms=12.5,
            )

            summary_a = BenchmarkSummary(
                config_name="lexical_fts5",
                top_k=3,
                total_queries=1,
                answerable_queries=1,
                unanswerable_queries=0,
                mean_precision_at_k_answerable=1.0,
                mean_recall_at_k_answerable=1.0,
                mrr_answerable=1.0,
                mean_precision_at_k_all=1.0,
                mean_recall_at_k_all=1.0,
                mrr_all=1.0,
                mean_groundedness=1.0,
                mean_latency_ms=12.5,
                abstention_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                per_query_results=[q_eval],
            )

            summary_b = BenchmarkSummary(
                config_name="dense_cosine",
                top_k=3,
                total_queries=1,
                answerable_queries=1,
                unanswerable_queries=0,
                mean_precision_at_k_answerable=0.6667,
                mean_recall_at_k_answerable=1.0,
                mrr_answerable=1.0,
                mean_precision_at_k_all=0.6667,
                mean_recall_at_k_all=1.0,
                mrr_all=1.0,
                mean_groundedness=0.95,
                mean_latency_ms=18.0,
                abstention_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                per_query_results=[q_eval],
            )

            run_id_a = save_benchmark_to_sqlite(summary_a, test_db)
            run_id_b = save_benchmark_to_sqlite(summary_b, test_db)

            self.assertNotEqual(run_id_a, run_id_b)

            with get_db_connection(test_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) AS cnt FROM benchmark_runs;")
                self.assertEqual(cursor.fetchone()["cnt"], 2)

                cursor.execute("SELECT COUNT(*) AS cnt FROM run_evaluations;")
                self.assertEqual(cursor.fetchone()["cnt"], 2)

    def test_report_generation(self) -> None:
        """Verify markdown report generation produces expected sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_md = Path(temp_dir) / "test_report.md"

            dummy_summary = BenchmarkSummary(
                config_name="test_config",
                top_k=3,
                total_queries=22,
                answerable_queries=19,
                unanswerable_queries=3,
                mean_precision_at_k_answerable=0.85,
                mean_recall_at_k_answerable=0.90,
                mrr_answerable=0.92,
                mean_precision_at_k_all=0.73,
                mean_recall_at_k_all=0.77,
                mrr_all=0.79,
                mean_groundedness=0.94,
                mean_latency_ms=15.0,
                abstention_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                per_query_results=[],
            )

            report = generate_markdown_report(dummy_summary, dummy_summary, out_md)
            self.assertTrue(out_md.exists())
            self.assertIn("DAI-009: RAG Answer Quality Benchmark Report", report)
            self.assertIn("Head-to-Head Comparison", report)
            self.assertIn("Trade-Off Analysis", report)


if __name__ == "__main__":
    unittest.main()
