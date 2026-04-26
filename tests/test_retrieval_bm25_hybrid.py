from __future__ import annotations

import unittest

from retrieval.bm25 import BM25Document, BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.semantic import SemanticResult


class TestRetrieval(unittest.TestCase):
    def test_bm25_search_returns_matching_doc(self) -> None:
        index = BM25Index(
            [
                BM25Document(doc_id="d1", text="cache strategy and ttl"),
                BM25Document(doc_id="d2", text="transformer embeddings"),
            ]
        )
        results = index.search("ttl cache", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].doc_id, "d1")

    def test_hybrid_search_validates_params(self) -> None:
        with self.assertRaises(ValueError):
            hybrid_search([], [], alpha=1.5)

    def test_hybrid_search_merges_ranks(self) -> None:
        semantic = [SemanticResult(doc_id="a", text="A", score=0.9, metadata={"source": "s1"})]
        bm25 = [type("BM25ResultLike", (), {"doc_id": "a", "text": "A", "score": 2.0, "metadata": {"source": "s1"}})()]
        merged = hybrid_search(semantic, bm25, top_k=1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].doc_id, "a")

    def test_bm25_returns_empty_for_empty_query(self) -> None:
        index = BM25Index([BM25Document(doc_id="d1", text="hello world")])
        self.assertEqual(index.search("", top_k=5), [])

    def test_hybrid_group_limit_backfills(self) -> None:
        semantic = [
            SemanticResult(doc_id="a", text="A", score=0.9, metadata={"source": "s1"}),
            SemanticResult(doc_id="b", text="B", score=0.8, metadata={"source": "s1"}),
            SemanticResult(doc_id="c", text="C", score=0.7, metadata={"source": "s2"}),
        ]
        bm25 = []
        merged = hybrid_search(semantic, bm25, top_k=2, max_per_group=1)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()

