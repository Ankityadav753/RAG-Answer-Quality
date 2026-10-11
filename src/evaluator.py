"""
Quantitative evaluation engine for DAI-009: RAG Answer Quality Benchmark.

Provides:
1. Standard Information Retrieval (IR) metrics: Precision@K, Recall@K, Reciprocal Rank (RR/MRR).
2. Answerability / Abstention metrics: Confusion matrix (TP, FP, FN, TN), Precision, Recall, F1.
3. Groundedness evaluation: Transparent heuristic token-containment between generated answer and context.
4. Structured evaluation containers and aggregate reporting models.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _normalize_token(token: str) -> str:
    """Normalize word tokens with simple suffix stemming."""
    t = token.lower()
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def extract_content_tokens(text: str) -> Set[str]:
    """
    Extract meaningful lexical content tokens from text, filtering stopwords.
    
    Args:
        text: Input string.
        
    Returns:
        Set[str]: Normalized content token set.
    """
    if not text:
        return set()

    stopwords = {
        "what", "is", "the", "for", "and", "in", "to", "of", "a", "an",
        "are", "do", "how", "on", "can", "i", "my", "if", "or", "which",
        "where", "when", "who", "whom", "will", "would", "at", "by", "from",
        "it", "this", "that", "these", "those", "there", "has", "have", "had",
        "be", "been", "being", "with", "as", "about", "must", "all", "each",
        "so", "no", "not", "any", "then", "into", "up", "out", "per"
    }

    words = re.findall(r"\b[A-Za-z0-9_-]+\b", text.lower())
    tokens: Set[str] = set()
    for w in words:
        parts = w.split("-")
        for p in parts:
            p_norm = _normalize_token(p)
            if len(p_norm) >= 2 and p_norm not in stopwords:
                tokens.add(p_norm)
    return tokens


# =============================================================================
# Retrieval Metrics (Precision@K, Recall@K, Reciprocal Rank)
# =============================================================================

def compute_precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Precision@K: Proportion of top-K retrieved items that are relevant.
    
    Convention:
    - If relevant_doc_ids is empty (unanswerable query), precision is 0.0.
    - If retrieved_doc_ids has fewer than k elements, evaluated against available items.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    if not relevant_doc_ids:
        return 0.0

    top_retrieved = list(retrieved_doc_ids[:k])
    if not top_retrieved:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_retrieved if doc_id in relevant_set)
    return round(hits / k, 4)


def compute_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K: Proportion of total ground-truth relevant documents found in top-K.
    
    Convention:
    - If relevant_doc_ids is empty (unanswerable query), recall is 0.0.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    if not relevant_doc_ids:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    top_retrieved_set = set(retrieved_doc_ids[:k])

    hits = len(relevant_set.intersection(top_retrieved_set))
    return round(hits / len(relevant_set), 4)


def compute_reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of the first relevant document in top-K.
    
    Returns:
        float: 1.0 / rank (1.0 for rank 1, 0.5 for rank 2, etc.), or 0.0 if not found.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    if not relevant_doc_ids or not retrieved_doc_ids:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)

    return 0.0


# =============================================================================
# Answer Groundedness & Abstention Metrics
# =============================================================================

def compute_groundedness_score(
    answer: str,
    context_chunks: Sequence[str],
    is_abstention: bool,
) -> float:
    """
    Calculate Groundedness score (Heuristic Token Containment).
    
    Measures the percentage of content words in the generated answer that directly
    appear in the retrieved context chunks.
    
    NOTE: This is a transparent rule-based lexical overlap proxy for faithfulness.
    It does NOT claim to be human-verified or an LLM-as-a-judge metric.
    
    Convention:
    - If the system correctly abstained, groundedness is 1.0 (no hallucinated claims).
    - If the answer has no content words, groundedness is 1.0.
    """
    if is_abstention:
        return 1.0

    if not answer or not answer.strip():
        return 1.0

    answer_tokens = extract_content_tokens(answer)
    if not answer_tokens:
        return 1.0

    combined_context = " ".join(context_chunks)
    context_tokens = extract_content_tokens(combined_context)

    supported_tokens = answer_tokens.intersection(context_tokens)
    return round(len(supported_tokens) / len(answer_tokens), 4)


def compute_abstention_metrics(
    predictions_is_abstention: Sequence[bool],
    ground_truth_is_answerable: Sequence[bool],
) -> Dict[str, float]:
    """
    Compute binary classification metrics for abstention behavior.
    
    Positive Class Definition:
    - Positive: Abstention (system abstains).
    - True Positive (TP): Ground truth is Unanswerable and system Abstained.
    - False Positive (FP): Ground truth is Answerable but system Abstained.
    - False Negative (FN): Ground truth is Unanswerable but system Attempted Answer.
    - True Negative (TN): Ground truth is Answerable and system Attempted Answer.
    """
    if len(predictions_is_abstention) != len(ground_truth_is_answerable):
        raise ValueError(
            f"Length mismatch: {len(predictions_is_abstention)} predictions vs "
            f"{len(ground_truth_is_answerable)} ground truth labels"
        )

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred_abstain, gt_answerable in zip(
        predictions_is_abstention, ground_truth_is_answerable
    ):
        gt_unanswerable = not gt_answerable
        if pred_abstain and gt_unanswerable:
            tp += 1
        elif pred_abstain and not gt_unanswerable:
            fp += 1
        elif not pred_abstain and gt_unanswerable:
            fn += 1
        else:
            tn += 1

    total = len(predictions_is_abstention)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
    f1 = (
        round(2.0 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = round((tp + tn) / total, 4) if total > 0 else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# =============================================================================
# Structured Evaluation Result Data Classes
# =============================================================================

@dataclass(frozen=True)
class PerQueryEvaluation:
    """Per-question evaluation record."""
    query_id: str
    query_text: str
    question_type: str
    is_answerable_ground_truth: bool
    relevant_doc_ids: List[str]
    retrieved_chunk_ids: List[str]
    retrieved_doc_ids: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    is_abstention: bool
    groundedness_score: float
    answer_text: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregated benchmark report for a specific retrieval configuration."""
    config_name: str
    top_k: int
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int
    # Metrics on answerable subset (ground truth exists)
    mean_precision_at_k_answerable: float
    mean_recall_at_k_answerable: float
    mrr_answerable: float
    # Metrics across all queries (including unanswerable where precision=0)
    mean_precision_at_k_all: float
    mean_recall_at_k_all: float
    mrr_all: float
    # Answer quality & timing
    mean_groundedness: float
    mean_latency_ms: float
    abstention_metrics: Dict[str, float]
    per_query_results: List[PerQueryEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
