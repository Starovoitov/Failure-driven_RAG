from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "ingestion" / "cleaner.py"
spec = importlib.util.spec_from_file_location("test_ingestion_cleaner_module", module_path)
assert spec is not None and spec.loader is not None
_cleaner_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_cleaner_mod)
cleanup_faiss_db = _cleaner_mod.cleanup_faiss_db


class TestIngestionCleaner(unittest.TestCase):
    def test_cleanup_index_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index_dir = root / "idx"
            index_dir.mkdir(parents=True)
            result = cleanup_faiss_db(
                persist_directory=str(root), index_name="idx", drop_persist_directory=False
            )
            self.assertTrue(result["index_deleted"])
            self.assertFalse(result["directory_deleted"])
            self.assertTrue(root.exists())

    def test_cleanup_full_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "idx").mkdir(parents=True)
            result = cleanup_faiss_db(
                persist_directory=str(root), index_name="idx", drop_persist_directory=True
            )
            self.assertTrue(result["directory_deleted"])

    def test_cleanup_when_paths_do_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "missing_root"
            result = cleanup_faiss_db(
                persist_directory=str(root), index_name="idx", drop_persist_directory=False
            )
            self.assertFalse(result["index_deleted"])
            self.assertFalse(result["directory_deleted"])

    def test_cleanup_drop_directory_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = cleanup_faiss_db(
                persist_directory=str(root), index_name="not-there", drop_persist_directory=True
            )
            self.assertFalse(result["index_deleted"])
            self.assertTrue(result["directory_deleted"])


if __name__ == "__main__":
    unittest.main()
