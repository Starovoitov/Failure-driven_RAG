from __future__ import annotations

import unittest

from evaluation.metrics import (
    evaluate_retrieval,
    RetrievalResult,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestEvaluationMetrics(unittest.TestCase):
    def test_basic_metrics(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["b", "x"]
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(hit_rate_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(hit_rate_at_k(retrieved, relevant, 2), 1.0)
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.5)

    def test_ndcg_handles_ideal_zero(self) -> None:
        self.assertEqual(ndcg_at_k(["a"], [], 1), 0.0)

    def test_retrieval_result_dataclass(self) -> None:
        item = RetrievalResult(query="q", retrieved_doc_ids=["a"], relevant_doc_ids=["a"])
        self.assertEqual(item.query, "q")

    def test_evaluate_retrieval_empty_results(self) -> None:
        out = evaluate_retrieval([], [1, 3])
        self.assertEqual(out["mrr"], 0.0)
        self.assertEqual(out["recall@1"], 0.0)
        self.assertEqual(out["ndcg@3"], 0.0)

    def test_precision_with_short_result_list(self) -> None:
        self.assertEqual(precision_at_k(["a"], ["a", "b"], 5), 1.0)


if __name__ == "__main__":
    unittest.main()

