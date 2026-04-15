from __future__ import annotations

from dataclasses import dataclass

from retrieval.bm25 import BM25Result
from retrieval.semantic import SemanticResult


@dataclass
class HybridResult:
    """Merged result from semantic + BM25 search."""

    doc_id: str
    text: str
    score: float
    semantic_score: float
    bm25_score: float
    metadata: dict


def hybrid_search(
    semantic_results: list[SemanticResult],
    bm25_results: list[BM25Result],
    alpha: float = 0.7,
    top_k: int = 5,
) -> list[HybridResult]:
    """
    Combine semantic and BM25 scores into one ranking.

    Score formula:
    - combined = alpha * semantic_score + (1 - alpha) * normalized_bm25_score

    `alpha` closer to 1.0 favors semantic similarity.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in range [0.0, 1.0]")

    bm25_max = max((item.score for item in bm25_results), default=0.0)
    bm25_norm = {
        item.doc_id: (item.score / bm25_max if bm25_max > 0 else 0.0) for item in bm25_results
    }
    semantic_map = {item.doc_id: item for item in semantic_results}
    bm25_map = {item.doc_id: item for item in bm25_results}

    all_doc_ids = set(semantic_map) | set(bm25_map)
    merged: list[HybridResult] = []
    for doc_id in all_doc_ids:
        semantic_item = semantic_map.get(doc_id)
        bm25_item = bm25_map.get(doc_id)

        semantic_score = semantic_item.score if semantic_item else 0.0
        bm25_score = bm25_norm.get(doc_id, 0.0)
        combined = alpha * semantic_score + (1.0 - alpha) * bm25_score

        source_item = semantic_item or bm25_item
        if source_item is None:
            continue

        merged.append(
            HybridResult(
                doc_id=doc_id,
                text=source_item.text,
                score=combined,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                metadata=source_item.metadata,
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[: max(top_k, 0)]
