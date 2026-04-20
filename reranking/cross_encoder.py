from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import CrossEncoder


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate passage before reranking."""

    doc_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedResult:
    """Reranked passage with cross-encoder score."""

    doc_id: str
    text: str
    score: float
    base_score: float
    metadata: dict[str, Any]


class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper.

    Usage:
    - retrieve initial candidates with BM25/semantic/hybrid,
    - call `rerank(query, candidates, top_k=...)` to reorder by CE relevance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        max_length: int = 512,
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not candidates or top_k <= 0:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            RerankedResult(
                doc_id=candidate.doc_id,
                text=candidate.text,
                score=float(score),
                base_score=float(candidate.score),
                metadata=candidate.metadata,
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
