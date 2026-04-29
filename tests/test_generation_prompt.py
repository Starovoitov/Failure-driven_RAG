from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "generation" / "prompt.py"
spec = importlib.util.spec_from_file_location("test_generation_prompt_module", module_path)
assert spec is not None and spec.loader is not None
_prompt_mod = importlib.util.module_from_spec(spec)
sys.modules["test_generation_prompt_module"] = _prompt_mod
spec.loader.exec_module(_prompt_mod)
SourceChunk = _prompt_mod.SourceChunk
build_rag_messages = _prompt_mod.build_rag_messages
format_context_with_citations = _prompt_mod.format_context_with_citations


class TestGenerationPrompt(unittest.TestCase):
    def test_format_context_with_citations(self) -> None:
        chunks = [
            SourceChunk(
                doc_id="d1", text="First document text", score=0.9, metadata={"title": "T1"}
            ),
            SourceChunk(
                doc_id="d2", text="Second document text", score=0.8, metadata={"url": "https://x"}
            ),
        ]
        context, used = format_context_with_citations(chunks, max_context_tokens=200)
        self.assertIn("[1]", context)
        self.assertIn("title=T1", context)
        self.assertEqual(len(used), 2)

    def test_build_rag_messages_uses_fallback_when_no_sources(self) -> None:
        payload = build_rag_messages(question="What?", chunks=[], top_k=3, max_context_tokens=100)
        self.assertIn("[no sources provided]", payload["user_prompt"])
        self.assertEqual(payload["used_chunks"], [])

    def test_merge_respects_top_k_by_score(self) -> None:
        chunks = [
            SourceChunk(doc_id="a", text="A", score=0.2),
            SourceChunk(doc_id="b", text="B", score=0.9),
            SourceChunk(doc_id="c", text="C", score=0.5),
        ]
        payload = build_rag_messages(question="Q", chunks=chunks, top_k=2, max_context_tokens=200)
        ids = [c.doc_id for c in payload["used_chunks"]]
        self.assertEqual(ids, ["b", "c"])

    def test_context_truncation_limits_selected_chunks(self) -> None:
        big_text = "x" * 800
        chunks = [
            SourceChunk(doc_id="d1", text=big_text, score=1.0),
            SourceChunk(doc_id="d2", text=big_text, score=0.5),
        ]
        payload = build_rag_messages(question="Q", chunks=chunks, top_k=2, max_context_tokens=50)
        self.assertTrue(payload["context_tokens_estimate"] <= 55)


if __name__ == "__main__":
    unittest.main()
