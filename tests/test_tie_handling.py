"""
Tests for benchmark report tie handling and winner determination (DAI-009 Phase 1).

Tests:
1. determine_winner correctly identifies ties for identical metric values.
2. determine_winner respects tolerance for floating-point values.
3. determine_winner handles higher_is_better=True vs False (e.g. latency).
4. generate_markdown_report outputs 'Tie' for equal groundedness scores.
"""

import tempfile
import unittest
from pathlib import Path

from src.benchmark_runner import determine_winner, generate_markdown_report
from src.evaluator import BenchmarkSummary


class TestTieHandling(unittest.TestCase):
    def test_determine_winner_exact_tie(self) -> None:
        """Verify identical scores return 'Tie'."""
        self.assertEqual(determine_winner(1.0, 1.0), "Tie")
        self.assertEqual(determine_winner(0.85, 0.85), "Tie")
        self.assertEqual(determine_winner(0.0, 0.0), "Tie")

    def test_determine_winner_tolerance_tie(self) -> None:
        """Verify scores within tolerance return 'Tie'."""
        self.assertEqual(determine_winner(0.99999, 1.00000, tolerance=1e-4), "Tie")
        self.assertEqual(determine_winner(1.201, 1.203, tolerance=0.01), "Tie")

    def test_determine_winner_higher_is_better(self) -> None:
        """Verify higher scores win when higher_is_better=True."""
        self.assertEqual(determine_winner(0.95, 0.85, higher_is_better=True), "Lexical")
        self.assertEqual(determine_winner(0.70, 0.88, higher_is_better=True), "Dense")

    def test_determine_winner_lower_is_better(self) -> None:
        """Verify lower scores win when higher_is_better=False (e.g. latency)."""
        self.assertEqual(determine_winner(1.5, 2.5, higher_is_better=False), "Lexical")
        self.assertEqual(determine_winner(3.0, 0.8, higher_is_better=False), "Dense")
        self.assertEqual(determine_winner(1.50, 1.51, higher_is_better=False, tolerance=0.05), "Tie")

    def test_markdown_report_groundedness_tie(self) -> None:
        """Verify markdown report displays 'Tie' when groundedness scores are equal."""
        dummy_abstention = {
            "tp": 1, "fp": 0, "fn": 2, "tn": 19,
            "precision": 1.0, "recall": 0.3333, "f1": 0.5, "accuracy": 0.9091,
        }

        lex_summary = BenchmarkSummary(
            config_name="lexical_fts5",
            top_k=3,
            total_queries=22,
            answerable_queries=19,
            unanswerable_queries=3,
            mean_precision_at_k_answerable=0.579,
            mean_recall_at_k_answerable=1.0,
            mrr_answerable=0.9737,
            mean_precision_at_k_all=0.5,
            mean_recall_at_k_all=0.8636,
            mrr_all=0.8409,
            mean_groundedness=1.0,
            mean_latency_ms=1.2,
            abstention_metrics=dummy_abstention,
            per_query_results=[],
        )

        dense_summary = BenchmarkSummary(
            config_name="dense_cosine",
            top_k=3,
            total_queries=22,
            answerable_queries=19,
            unanswerable_queries=3,
            mean_precision_at_k_answerable=0.2105,
            mean_recall_at_k_answerable=0.5789,
            mrr_answerable=0.4561,
            mean_precision_at_k_all=0.1818,
            mean_recall_at_k_all=0.5,
            mrr_all=0.3939,
            mean_groundedness=1.0,
            mean_latency_ms=0.5,
            abstention_metrics=dummy_abstention,
            per_query_results=[],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_report.md"
            content = generate_markdown_report(lex_summary, dense_summary, out_file)

            # Groundedness is 1.0 for both: must be reported as Tie
            self.assertIn("| **Answer Quality** | **Groundedness Score** (Heuristic) | **1.0000** | **1.0000** | Tie |", content)
            self.assertNotIn("| **Answer Quality** | **Groundedness Score** (Heuristic) | **1.0000** | **1.0000** | Lexical |", content)


if __name__ == "__main__":
    unittest.main()
