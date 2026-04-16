from __future__ import annotations

import shutil
from pathlib import Path

import chromadb


def cleanup_chroma_db(
    persist_directory: str = "data/chroma",
    collection_name: str = "rag_chunks",
    drop_persist_directory: bool = False,
) -> dict[str, bool]:
    """
    Clean up Chroma DB by deleting a collection and optionally removing DB files.

    Returns status flags describing what was removed.
    """
    db_path = Path(persist_directory)
    collection_deleted = False
    directory_deleted = False

    if db_path.exists():
        client = chromadb.PersistentClient(path=str(db_path))
        try:
            client.delete_collection(name=collection_name)
            collection_deleted = True
        except Exception:  # noqa: BLE001
            # No-op if collection does not exist or cannot be deleted.
            collection_deleted = False

    if drop_persist_directory and db_path.exists():
        shutil.rmtree(db_path)
        directory_deleted = True

    return {
        "collection_deleted": collection_deleted,
        "directory_deleted": directory_deleted,
    }
