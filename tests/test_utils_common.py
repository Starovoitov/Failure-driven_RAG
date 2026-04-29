from __future__ import annotations

import unittest

from utils.common import min_max_normalize, rank_weight, tokenize


class TestUtilsCommon(unittest.TestCase):
    def test_tokenize_default_and_bm25_modes(self) -> None:
        text = "Hello, world!"
        self.assertEqual(tokenize(text), ["Hello", ",", "world", "!"])
        self.assertEqual(tokenize(text, for_bm25=True), ["hello", "world"])

    def test_min_max_normalize(self) -> None:
        self.assertEqual(min_max_normalize({}), {})
        self.assertEqual(min_max_normalize({"a": 2.0, "b": 2.0}), {"a": 0.5, "b": 0.5})
        self.assertEqual(min_max_normalize({"a": 1.0, "b": 3.0}), {"a": 0.0, "b": 1.0})

    def test_rank_weight(self) -> None:
        self.assertEqual(rank_weight(1), 1.0)
        self.assertEqual(rank_weight(10), 0.7)
        self.assertEqual(rank_weight(40), 0.4)

    def test_min_max_normalize_preserves_keys(self) -> None:
        src = {"x": 2.0, "y": 4.0, "z": 3.0}
        out = min_max_normalize(src)
        self.assertEqual(set(out.keys()), set(src.keys()))
