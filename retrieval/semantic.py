from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class SemanticDocument:
    """Document container for semantic search."""

    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticResult:
    """Returned item for semantic retrieval."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value in [-1.0, 1.0].
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")
    if not vector_a:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def search_semantic(
    query_embedding: list[float],
    documents: list[SemanticDocument],
    top_k: int = 5,
    min_score: float | None = None,
) -> list[SemanticResult]:
    """
    Rank documents by cosine similarity to the query embedding.

    - `query_embedding`: embedding for the user query
    - `documents`: corpus with precomputed embeddings
    - `top_k`: how many best results to return
    - `min_score`: optional similarity threshold
    """
    scored: list[SemanticResult] = []
    for doc in documents:
        score = cosine_similarity(query_embedding, doc.embedding)
        if min_score is not None and score < min_score:
            continue
        scored.append(
            SemanticResult(
                doc_id=doc.doc_id,
                text=doc.text,
                score=score,
                metadata=doc.metadata,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(top_k, 0)]
