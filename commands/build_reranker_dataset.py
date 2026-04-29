from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from ingestion.loaders import load_chunk_texts
from utils.common import rank_weight


class FailureSampleFields(BaseModel):
    query: str
    positives: list[str]
    retrieved: list[str]
    retrieved_full: list[str]
    bm25_branch: list[str]
    bucket: str
    source_miss_type: str


class RerankerContext(BaseModel):
    schema_version: str = "reranker_context_v1"
    query: str
    positives: list[str]
    negatives: list[str]
    weights: dict[str, float]
    failure_bucket: str
    source_miss_type: str


class BuildContextsStats(BaseModel):
    samples_seen: int = 0
    samples_used: int = 0
    contexts_written: int = 0
    missing_positive_text: int = 0
    missing_negative_text: int = 0
    contexts_ranking_cutoff_failure: int = 0
    contexts_true_recall_failure: int = 0
    contexts_other: int = 0


def _stats_inc(stats: BuildContextsStats | dict[str, int], key: str) -> None:
    if isinstance(stats, dict):
        stats[key] = int(stats.get(key, 0)) + 1
        return
    setattr(stats, key, int(getattr(stats, key, 0)) + 1)


def _extract_failure_sample_fields(sample: dict) -> FailureSampleFields:
    return FailureSampleFields(
        query=str(sample.get("query", "")).strip(),
        positives=[str(doc_id) for doc_id in sample.get("relevant_doc_ids", [])],
        retrieved=[str(doc_id) for doc_id in sample.get("retrieved_top_k_doc_ids", [])],
        retrieved_full=[str(doc_id) for doc_id in sample.get("retrieved_full_doc_ids", [])],
        bm25_branch=[str(doc_id) for doc_id in sample.get("bm25_branch_doc_ids", [])],
        bucket=str(sample.get("bucket", "")),
        source_miss_type=str(sample.get("source_miss_type", "")),
    )


def _sample_weight_for_bucket(
    bucket: str,
    *,
    ranking_cutoff_weight: float,
    true_recall_weight: float,
    default_weight: float,
) -> float:
    if bucket == "ranking_cutoff_failure":
        return ranking_cutoff_weight
    if bucket == "true_recall_failure":
        return true_recall_weight
    return default_weight


def _collect_positive_ids(
    positives: list[str],
    chunk_texts: dict[str, str],
    stats: BuildContextsStats | dict[str, int],
) -> list[str]:
    positive_ids: list[str] = []
    for positive_id in positives:
        if positive_id not in chunk_texts:
            _stats_inc(stats, "missing_positive_text")
            continue
        positive_ids.append(positive_id)
    return positive_ids


def _build_negative_pool(
    *,
    bucket: str,
    retrieved: list[str],
    retrieved_full: list[str],
    bm25_branch: list[str],
    max_negative_rank: int,
) -> list[str]:
    if bucket == "ranking_cutoff_failure":
        return retrieved[:max_negative_rank]
    if bucket == "true_recall_failure":
        return (bm25_branch or retrieved_full or retrieved)[:max_negative_rank]
    return retrieved[:max_negative_rank]


def _collect_negative_ids(
    *,
    negative_pool: list[str],
    retrieved_full: list[str],
    positive_ids: list[str],
    source_miss_type: str,
    chunk_texts: dict[str, str],
    sample_weight: float,
    max_negatives: int,
    stats: BuildContextsStats | dict[str, int],
) -> tuple[list[str], dict[str, float]]:
    negative_ids: list[str] = []
    negative_weights: dict[str, float] = {}
    positive_set = set(positive_ids)

    for rank, negative_id in enumerate(negative_pool, start=1):
        if len(negative_ids) >= max_negatives:
            break
        if negative_id in positive_set:
            continue
        if negative_id not in chunk_texts:
            _stats_inc(stats, "missing_negative_text")
            continue
        if negative_id in negative_weights:
            continue
        negative_ids.append(negative_id)
        negative_weights[negative_id] = sample_weight * rank_weight(rank)

    if source_miss_type in {"embedding_miss", "bm25_miss"} and len(negative_ids) < max_negatives:
        for rank, negative_id in enumerate(retrieved_full, start=len(negative_ids) + 1):
            if len(negative_ids) >= max_negatives:
                break
            if negative_id in positive_set or negative_id in negative_weights:
                continue
            if negative_id not in chunk_texts:
                continue
            negative_ids.append(negative_id)
            negative_weights[negative_id] = sample_weight * rank_weight(rank)

    return negative_ids, negative_weights


def _update_context_bucket_stats(stats: BuildContextsStats, bucket: str) -> None:
    stats.contexts_written += 1
    if bucket == "ranking_cutoff_failure":
        stats.contexts_ranking_cutoff_failure += 1
    elif bucket == "true_recall_failure":
        stats.contexts_true_recall_failure += 1
    else:
        stats.contexts_other += 1


def build_contexts(
    *,
    report: dict,
    chunk_texts: dict[str, str],
    max_negative_rank: int,
    max_negatives: int,
    ranking_cutoff_weight: float,
    true_recall_weight: float,
    default_weight: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    evaluation = report.get("evaluation", {})
    diagnostics = report.get("diagnostics", {})
    failure_analysis = diagnostics.get("failure_analysis", {})
    samples = evaluation.get("failed_queries_for_manual_inspection", [])
    if not samples:
        samples = failure_analysis.get("manual_inspection_samples", [])

    contexts: list[RerankerContext] = []
    stats = BuildContextsStats()

    for sample in samples:
        stats.samples_seen += 1
        fields = _extract_failure_sample_fields(sample)
        query = fields.query
        positives = fields.positives
        retrieved = fields.retrieved
        retrieved_full = fields.retrieved_full
        bm25_branch = fields.bm25_branch
        bucket = fields.bucket
        source_miss_type = fields.source_miss_type
        sample_weight = _sample_weight_for_bucket(
            bucket,
            ranking_cutoff_weight=ranking_cutoff_weight,
            true_recall_weight=true_recall_weight,
            default_weight=default_weight,
        )

        if not query or not positives or not retrieved:
            continue

        stats.samples_used += 1
        positive_ids = _collect_positive_ids(positives, chunk_texts, stats)
        negative_pool = _build_negative_pool(
            bucket=bucket,
            retrieved=retrieved,
            retrieved_full=retrieved_full,
            bm25_branch=bm25_branch,
            max_negative_rank=max_negative_rank,
        )
        negative_ids, negative_weights = _collect_negative_ids(
            negative_pool=negative_pool,
            retrieved_full=retrieved_full,
            positive_ids=positive_ids,
            source_miss_type=source_miss_type,
            chunk_texts=chunk_texts,
            sample_weight=sample_weight,
            max_negatives=max_negatives,
            stats=stats,
        )

        if not positive_ids or not negative_ids:
            continue

        contexts.append(
            RerankerContext(
                query=query,
                positives=positive_ids,
                negatives=negative_ids,
                weights=negative_weights,
                failure_bucket=bucket,
                source_miss_type=source_miss_type,
            )
        )
        _update_context_bucket_stats(stats, bucket)

    return [item.model_dump() for item in contexts], stats.model_dump()

