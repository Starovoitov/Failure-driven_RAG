from __future__ import annotations

import sys
import types
import unittest


class _DummyCrossEncoder:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def predict(self, pairs, batch_size=32):  # noqa: ARG002
        return [0.1 + (0.01 * i) for i, _ in enumerate(pairs)]


class TestRerankingCrossEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_st = sys.modules.get("sentence_transformers")
        fake = types.ModuleType("sentence_transformers")
        fake.CrossEncoder = _DummyCrossEncoder
        sys.modules["sentence_transformers"] = fake

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old_st is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = cls._old_st

    def test_calibrate_ce_scores_minmax(self) -> None:
        from reranking.cross_encoder import CEScoreCalibrationMode, calibrate_ce_scores

        out = calibrate_ce_scores([1.0, 3.0], CEScoreCalibrationMode.MINMAX, temperature=1.0)
        self.assertEqual(out, [0.0, 1.0])

    def test_rerank_returns_ordered_results(self) -> None:
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        rr = CrossEncoderReranker(model_name="dummy")
        results = rr.rerank(
            query="q",
            candidates=[RerankCandidate(doc_id="a", text="alpha", score=1.0)],
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "a")

    def test_rerank_returns_empty_for_non_positive_top_k(self) -> None:
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        rr = CrossEncoderReranker(model_name="dummy")
        results = rr.rerank(
            query="q", candidates=[RerankCandidate(doc_id="a", text="alpha")], top_k=0
        )
        self.assertEqual(results, [])

    def test_rerank_filters_blank_text_candidates(self) -> None:
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        rr = CrossEncoderReranker(model_name="dummy")
        results = rr.rerank(
            query="q",
            candidates=[
                RerankCandidate(doc_id="a", text="   "),
                RerankCandidate(doc_id="b", text="beta"),
            ],
            top_k=5,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "b")


if __name__ == "__main__":
    unittest.main()
