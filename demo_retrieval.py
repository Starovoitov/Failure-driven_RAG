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

from retrieval.bm25 import BM25Document, BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.semantic import SemanticDocument, search_semantic

# Same model as embeddings/embedder.py (E5 expects query:/passage: prefixes).
DEFAULT_MODEL = "intfloat/e5-small-v2"

SAMPLE_DOCS: list[tuple[str, str]] = [
    (
        "1",
        "Redis is an in-memory data store often used for caching and session storage.",
    ),
    (
        "2",
        "PostgreSQL is a relational database; it persists data to disk with ACID guarantees.",
    ),
    (
        "3",
        "Vector databases store embeddings for semantic search and nearest-neighbor queries.",
    ),
    (
        "4",
        "HTTP caching uses headers like Cache-Control to avoid redundant network requests.",
    ),
]


def run_demo(query: str, top_k: int, model_name: str) -> None:
    bm25_docs = [BM25Document(doc_id=doc_id, text=text) for doc_id, text in SAMPLE_DOCS]
    bm25_index = BM25Index(bm25_docs)
    bm25_results = bm25_index.search(query, top_k=top_k)

    model = SentenceTransformer(model_name)
    query_vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    passage_inputs = [f"passage: {text}" for _, text in SAMPLE_DOCS]
    passage_vecs = model.encode(
        passage_inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    semantic_docs = [
        SemanticDocument(
            doc_id=doc_id,
            text=text,
            embedding=passage_vecs[i].tolist(),
        )
        for i, (doc_id, text) in enumerate(SAMPLE_DOCS)
    ]
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
    args = parser.parse_args()
    run_demo(query=args.query, top_k=args.top_k, model_name=args.model)


if __name__ == "__main__":
    main()
