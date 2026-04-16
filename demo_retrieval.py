#!/usr/bin/env python3
"""
Run BM25, semantic (cosine), and hybrid retrieval on a tiny sample corpus.

Usage from project root:
  poetry run python demo_retrieval.py
  poetry run python demo_retrieval.py --query "how does caching work"
"""

from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from ingestion.loaders import (
    load_bm25_documents_from_dataset,
    load_semantic_documents_from_chroma,
)
from retrieval.bm25 import BM25Document, BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.semantic import search_semantic

# Same model as embeddings/embedder.py (E5 expects query:/passage: prefixes).
DEFAULT_MODEL = "intfloat/e5-small-v2"


def run_demo(
    query: str,
    top_k: int,
    model_name: str,
    dataset_path: str,
    chroma_path: str,
    collection_name: str,
) -> None:
    dataset_docs = load_bm25_documents_from_dataset(dataset_path=dataset_path)
    if not dataset_docs:
        raise ValueError(
            f"No raw_chunk records found in {dataset_path}. "
            "Run parser ingestion before demo retrieval."
        )

    bm25_docs = [
        BM25Document(doc_id=item["id"], text=item["text"], metadata=item["metadata"])
        for item in dataset_docs
    ]
    bm25_index = BM25Index(bm25_docs)
    bm25_results = bm25_index.search(query, top_k=top_k)

    model = SentenceTransformer(model_name)
    query_vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    semantic_docs = load_semantic_documents_from_chroma(
        persist_directory=chroma_path,
        collection_name=collection_name,
    )
    if not semantic_docs:
        raise ValueError(
            f"No embeddings found in Chroma collection '{collection_name}' at {chroma_path}. "
            "Run parser + embedding ingestion first."
        )
    semantic_results = search_semantic(query_vec, semantic_docs, top_k=top_k)

    hybrid_results = hybrid_search(semantic_results, bm25_results, alpha=0.7, top_k=top_k)

    def print_block(title: str, rows: list) -> None:
        print(f"\n--- {title} ---")
        if not rows:
            print("(no results)")
            return
        for r in rows:
            text_preview = (r.text[:120] + "…") if len(r.text) > 120 else r.text
            if hasattr(r, "semantic_score"):
                print(
                    f"  id={r.doc_id}  score={r.score:.4f}  "
                    f"sem={r.semantic_score:.4f}  bm25_norm={r.bm25_score:.4f}"
                )
            else:
                print(f"  id={r.doc_id}  score={r.score:.4f}")
            print(f"       {text_preview}")

    print_block("BM25 (lexical)", bm25_results)
    print_block("Semantic (cosine)", semantic_results)
    print_block("Hybrid (0.7 * semantic + 0.3 * bm25_norm)", hybrid_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo BM25 + semantic + hybrid retrieval.")
    parser.add_argument(
        "--query",
        "-q",
        default="database caching performance",
        help="Search query text.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=4,
        help="Number of hits to show per method.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_dataset.jsonl",
        help="Dataset JSONL created by parser pipeline.",
    )
    parser.add_argument(
        "--chroma-path",
        default="data/chroma",
        help="Persistent Chroma directory containing chunk embeddings.",
    )
    parser.add_argument(
        "--collection",
        default="rag_chunks",
        help="Chroma collection name with precomputed embeddings.",
    )
    args = parser.parse_args()
    run_demo(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        dataset_path=args.dataset,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
