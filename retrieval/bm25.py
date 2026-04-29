from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from utils.common import tokenize


class BM25Document(BaseModel):
    """Document container for BM25 lexical search."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BM25Result(BaseModel):
    """Returned item for BM25 retrieval."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class BM25Index:
    """
    In-memory BM25 index.

    Defaults:
    - `k1=1.5`: term frequency saturation
    - `b=0.75`: document length normalization
    """

    def __init__(self, documents: list[BM25Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.avg_doc_length = 0.0
        self._build()

    def _build(self) -> None:
        total_length = 0
        for doc in self.documents:
            tokens = tokenize(doc.text, for_bm25=True)
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            self.doc_term_freqs.append(term_freqs)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            for term in term_freqs:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        if self.documents:
            self.avg_doc_length = total_length / len(self.documents)

    def _idf(self, term: str) -> float:
        """BM25 IDF with smoothing."""
        doc_count = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        """Rank documents by BM25 score for query text."""
        query_terms = tokenize(query, for_bm25=True)
        if not query_terms or not self.documents:
            return []

        scores = [0.0 for _ in self.documents]
        for term in query_terms:
            idf = self._idf(term)
            for index, term_freqs in enumerate(self.doc_term_freqs):
                term_frequency = term_freqs.get(term, 0)
                if term_frequency == 0:
                    continue

                doc_length = self.doc_lengths[index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * (doc_length / (self.avg_doc_length or 1.0))
                )
                scores[index] += idf * (term_frequency * (self.k1 + 1)) / denominator

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results: list[BM25Result] = []
        for idx in ranked_indices[: max(top_k, 0)]:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(
                BM25Result(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=scores[idx],
                    metadata=doc.metadata,
                )
            )
        return results
