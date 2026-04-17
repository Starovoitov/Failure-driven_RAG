from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from retrieval.semantic import SemanticDocument


INDEX_FILENAME = "vectors.index"
STORE_FILENAME = "store.json"


def _persist_paths(persist_directory: str, index_name: str) -> Path:
    root = Path(persist_directory) / index_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_faiss_index(
    embedding_records: list[dict[str, Any]],
    persist_directory: str = "data/faiss",
    index_name: str = "rag_chunks",
) -> int:
    """
    Build a FAISS IndexFlatIP index from normalized embedding vectors and persist.

    Vectors must be L2-normalized so inner product equals cosine similarity.
    Row i in FAISS matches store.json entries at index i.
    """
    root = _persist_paths(persist_directory, index_name)
    index_path = root / INDEX_FILENAME
    store_path = root / STORE_FILENAME

    if not embedding_records:
        store = {"ids": [], "texts": [], "metadatas": []}
        store_path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        if index_path.exists():
            index_path.unlink()
        return 0

    dim = len(embedding_records[0]["embedding"])
    vectors = np.array(
        [r["embedding"] for r in embedding_records],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_path))

    store = {
        "ids": [r["id"] for r in embedding_records],
        "texts": [r["text"] for r in embedding_records],
        "metadatas": [r.get("metadata", {}) for r in embedding_records],
    }
    store_path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    return len(embedding_records)


def load_semantic_documents_from_faiss(
    persist_directory: str = "data/faiss",
    index_name: str = "rag_chunks",
) -> list[SemanticDocument]:
    """Load vectors and parallel text/metadata from a persisted FAISS index."""
    root = Path(persist_directory) / index_name
    index_path = root / INDEX_FILENAME
    store_path = root / STORE_FILENAME

    if not store_path.is_file():
        return []

    store = json.loads(store_path.read_text(encoding="utf-8"))
    ids: list[str] = store.get("ids", [])
    texts: list[str] = store.get("texts", [])
    metadatas: list[Any] = store.get("metadatas", [])

    embeddings_flat: list[list[float]] = []
    if index_path.is_file():
        index = faiss.read_index(str(index_path))
        n = int(index.ntotal)
        if n > 0:
            full = index.reconstruct_n(0, n)
            embeddings_flat = full.tolist()

    results: list[SemanticDocument] = []
    for i, doc_id in enumerate(ids):
        text = texts[i] if i < len(texts) else ""
        embedding = embeddings_flat[i] if i < len(embeddings_flat) else []
        metadata = metadatas[i] if i < len(metadatas) else {}
        if text == "" or len(embedding) == 0:
            continue
        md = metadata if isinstance(metadata, dict) else {}
        results.append(
            SemanticDocument(
                doc_id=doc_id,
                text=text,
                embedding=list(embedding),
                metadata=md,
            )
        )
    return results
