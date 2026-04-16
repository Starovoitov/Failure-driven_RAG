from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import chromadb

from embeddings.embedder import generate_embeddings, upsert_embeddings_to_chroma
from parser.pipeline import run_pipeline
from retrieval.semantic import SemanticDocument


def _read_raw_chunks(dataset_path: str) -> Iterator[dict[str, Any]]:
    """Stream raw chunk records from dataset JSONL."""
    with Path(dataset_path).open("r", encoding="utf-8") as dataset:
        for line in dataset:
            item = json.loads(line)
            if item.get("record_type") != "raw_chunk":
                continue
            yield item


def run_parser_and_upsert_to_chroma(
    dataset_path: str = "data/rag_dataset.jsonl",
    persist_directory: str = "data/chroma",
    collection_name: str = "rag_chunks",
    model_name: str = "intfloat/e5-small-v2",
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
) -> dict[str, int]:
    """
    Run parser pipeline and upsert chunk embeddings into Chroma.

    Returns parser stats with additional embedding/chroma counters.
    """
    stats = run_pipeline(
        output_path=dataset_path,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
    )
    embedding_input_path = Path(dataset_path).with_name("embeddings_input.jsonl")
    raw_chunks_count = 0
    with embedding_input_path.open("w", encoding="utf-8") as out:
        for row in _read_raw_chunks(dataset_path):
            payload = {
                "id": row["chunk_id"],
                "text": row["text"],
                "metadata": row.get("metadata", {}),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            raw_chunks_count += 1

    records = generate_embeddings(
        input_jsonl=str(embedding_input_path),
        model_name=model_name,
    )
    upserted = upsert_embeddings_to_chroma(
        embedding_records=records,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    stats["raw_chunks_for_embedding"] = raw_chunks_count
    stats["embeddings_upserted"] = upserted
    return stats


def load_bm25_documents_from_dataset(dataset_path: str = "data/rag_dataset.jsonl") -> list[dict[str, Any]]:
    """Load raw chunk documents from dataset for lexical retrieval."""
    return [
        {
            "id": item["chunk_id"],
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        }
        for item in _read_raw_chunks(dataset_path)
    ]


def load_semantic_documents_from_chroma(
    persist_directory: str = "data/chroma",
    collection_name: str = "rag_chunks",
) -> list[SemanticDocument]:
    """Load documents and precomputed embeddings from Chroma."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    rows = collection.get(include=["documents", "embeddings", "metadatas"])

    ids = rows.get("ids", [])
    documents = rows.get("documents", [])
    embeddings = rows.get("embeddings", [])
    metadatas = rows.get("metadatas", [])

    results: list[SemanticDocument] = []
    for idx, doc_id in enumerate(ids):
        text = documents[idx] if idx < len(documents) else ""
        embedding = embeddings[idx] if idx < len(embeddings) else []
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if text is None or text == "":
            continue
        if embedding is None or len(embedding) == 0:
            continue
        results.append(
            SemanticDocument(
                doc_id=doc_id,
                text=text,
                embedding=list(embedding),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return results
