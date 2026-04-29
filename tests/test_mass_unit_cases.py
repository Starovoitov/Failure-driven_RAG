from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "generation" / "prompt.py"
spec = importlib.util.spec_from_file_location("test_mass_generation_prompt_module", module_path)
assert spec is not None and spec.loader is not None
_prompt_mod = importlib.util.module_from_spec(spec)
sys.modules["test_mass_generation_prompt_module"] = _prompt_mod
spec.loader.exec_module(_prompt_mod)
SourceChunk = _prompt_mod.SourceChunk
estimate_tokens = _prompt_mod.estimate_tokens
merge_top_k_documents = _prompt_mod.merge_top_k_documents
format_context_with_citations = _prompt_mod.format_context_with_citations
build_rag_messages = _prompt_mod.build_rag_messages


class TestUtilsCommonMass(unittest.TestCase):
    pass


class TestGenerationPromptMass(unittest.TestCase):
    pass


def _add_test(cls: type[unittest.TestCase], name: str, fn) -> None:
    setattr(cls, name, fn)
