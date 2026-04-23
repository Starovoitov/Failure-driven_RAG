#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_chunk_texts(rag_dataset_path: Path) -> dict[str, str]:
    chunk_texts: dict[str, str] = {}
    with rag_dataset_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if row.get("record_type") != "raw_chunk":
                continue
            chunk_id = str(row.get("chunk_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if chunk_id and text:
                chunk_texts[chunk_id] = text
    return chunk_texts


def _rank_weight(rank: int) -> float:
    # Rank-aware contrastive weighting:
    # - 1..5: hardest negatives, full weight
    # - 6..15: medium-hard negatives
    # - 16..50: tail negatives
    if rank <= 5:
        return 1.0
    if rank <= 15:
        return 0.7
    return 0.4


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
    diagnostics = report.get("diagnostics", {})
    failure_analysis = diagnostics.get("failure_analysis", {})
    samples = failure_analysis.get("manual_inspection_samples", [])

    contexts: list[dict[str, object]] = []
    stats = {
        "samples_seen": 0,
        "samples_used": 0,
        "contexts_written": 0,
        "missing_positive_text": 0,
        "missing_negative_text": 0,
        "contexts_ranking_cutoff_failure": 0,
        "contexts_true_recall_failure": 0,
        "contexts_other": 0,
    }

    for sample in samples:
        stats["samples_seen"] += 1
        query = str(sample.get("query", "")).strip()
        positives = [str(doc_id) for doc_id in sample.get("relevant_doc_ids", [])]
        retrieved = [str(doc_id) for doc_id in sample.get("retrieved_top_k_doc_ids", [])]
        retrieved_full = [str(doc_id) for doc_id in sample.get("retrieved_full_doc_ids", [])]
        bm25_branch = [str(doc_id) for doc_id in sample.get("bm25_branch_doc_ids", [])]
        bucket = str(sample.get("bucket", ""))
        source_miss_type = str(sample.get("source_miss_type", ""))
        if bucket == "ranking_cutoff_failure":
            sample_weight = ranking_cutoff_weight
        elif bucket == "true_recall_failure":
            sample_weight = true_recall_weight
        else:
            sample_weight = default_weight

        if not query or not positives or not retrieved:
            continue

        stats["samples_used"] += 1
        positive_ids: list[str] = []
        for positive_id in positives:
            if positive_id not in chunk_texts:
                stats["missing_positive_text"] += 1
                continue
            positive_ids.append(positive_id)

        if bucket == "ranking_cutoff_failure":
            negative_pool = retrieved[:max_negative_rank]
        elif bucket == "true_recall_failure":
            negative_pool = (bm25_branch or retrieved_full or retrieved)[:max_negative_rank]
        else:
            negative_pool = retrieved[:max_negative_rank]

        negative_ids: list[str] = []
        negative_weights: dict[str, float] = {}
        positive_set = set(positive_ids)
        for rank, negative_id in enumerate(negative_pool, start=1):
            if len(negative_ids) >= max_negatives:
                break
            if negative_id in positive_set:
                continue
            if negative_id not in chunk_texts:
                stats["missing_negative_text"] += 1
                continue
            if negative_id in negative_weights:
                continue
            negative_ids.append(negative_id)
            negative_weights[negative_id] = sample_weight * _rank_weight(rank)

        if source_miss_type in {"embedding_miss", "bm25_miss"} and len(negative_ids) < max_negatives:
            for rank, negative_id in enumerate(retrieved_full, start=len(negative_ids) + 1):
                if len(negative_ids) >= max_negatives:
                    break
                if negative_id in positive_set or negative_id in negative_weights:
                    continue
                if negative_id not in chunk_texts:
                    continue
                negative_ids.append(negative_id)
                negative_weights[negative_id] = sample_weight * _rank_weight(rank)

        if not positive_ids or not negative_ids:
            continue

        contexts.append(
            {
                "schema_version": "reranker_context_v1",
                "query": query,
                "positives": positive_ids,
                "negatives": negative_ids,
                "weights": negative_weights,
                "failure_bucket": bucket,
                "source_miss_type": source_miss_type,
            }
        )
        stats["contexts_written"] += 1
        if bucket == "ranking_cutoff_failure":
            stats["contexts_ranking_cutoff_failure"] += 1
        elif bucket == "true_recall_failure":
            stats["contexts_true_recall_failure"] += 1
        else:
            stats["contexts_other"] += 1

    return contexts, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context-aware reranker training data from retrieval report.")
    parser.add_argument("--eval-report", type=Path, default=Path("data/retrieval_report_best.json"))
    parser.add_argument("--rag-dataset", type=Path, default=Path("data/rag_dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/reranker_train.jsonl"))
    parser.add_argument("--max-negative-rank", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=16)
    parser.add_argument("--ranking-cutoff-weight", type=float, default=2.0)
    parser.add_argument("--true-recall-weight", type=float, default=0.3)
    parser.add_argument("--default-weight", type=float, default=1.0)
    args = parser.parse_args()

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    chunk_texts = load_chunk_texts(args.rag_dataset)
    contexts, stats = build_contexts(
        report=report,
        chunk_texts=chunk_texts,
        max_negative_rank=max(1, args.max_negative_rank),
        max_negatives=max(1, args.max_negatives),
        ranking_cutoff_weight=max(0.1, args.ranking_cutoff_weight),
        true_recall_weight=max(0.1, args.true_recall_weight),
        default_weight=max(0.1, args.default_weight),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for row in contexts:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out} ({len(contexts)} contexts).")
    print("Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
