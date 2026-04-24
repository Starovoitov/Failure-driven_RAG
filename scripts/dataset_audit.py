#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_raw_chunks(rag_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with rag_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if row.get("record_type") == "raw_chunk":
                chunks.append(row)
    return chunks


def load_eval_rows(eval_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            rows.append(json.loads(line))
    return rows


def top_share(counter: Counter[str], top_k: int) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(v for _, v in counter.most_common(top_k)) / total


def _build_chunk_level_stats(raw_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ids = {str(item.get("chunk_id")) for item in raw_chunks}
    token_counts = [int(item.get("token_count", 0)) for item in raw_chunks]
    overlap_tokens = [int(item.get("overlap_tokens", 0)) for item in raw_chunks]
    categories = Counter((item.get("metadata") or {}).get("category", "") for item in raw_chunks)
    urls = Counter((item.get("metadata") or {}).get("url", "") for item in raw_chunks)
    text_counter = Counter((item.get("text") or "").strip() for item in raw_chunks)
    duplicate_text_entries = sum(1 for _, cnt in text_counter.items() if cnt > 1)
    chunk_to_url = {str(item.get("chunk_id")): (item.get("metadata") or {}).get("url", "") for item in raw_chunks}
    return {
        "chunk_ids": chunk_ids,
        "token_counts": token_counts,
        "overlap_tokens": overlap_tokens,
        "categories": categories,
        "urls": urls,
        "duplicate_text_entries": duplicate_text_entries,
        "chunk_to_url": chunk_to_url,
    }


def _collect_evaluation_counters(
    eval_rows: list[dict[str, Any]],
    chunk_ids: set[str],
) -> dict[str, Any]:
    evidence_counts: list[int] = []
    resolution = Counter()
    gt_chunk_counter = Counter()
    unknown_chunk_refs = 0

    for row in eval_rows:
        expected = row.get("expected_evidence") or {}
        ids = [str(x) for x in (expected.get("chunk_ids") or []) if x is not None]
        evidence_counts.append(len(ids))
        resolution[str(expected.get("resolution_method"))] += 1
        for cid in ids:
            gt_chunk_counter[cid] += 1
            if cid not in chunk_ids:
                unknown_chunk_refs += 1

    return {
        "evidence_counts": evidence_counts,
        "resolution": resolution,
        "gt_chunk_counter": gt_chunk_counter,
        "unknown_chunk_refs": unknown_chunk_refs,
    }


def _build_gt_url_counter(gt_chunk_counter: Counter[str], chunk_to_url: dict[str, str]) -> Counter[str]:
    gt_url_counter = Counter()
    for cid, count in gt_chunk_counter.items():
        gt_url_counter[chunk_to_url.get(cid, "")] += count
    return gt_url_counter


def _quality_score(
    *,
    rows_total: int,
    queries_with_no_gt: int,
    gt_url_counter: Counter[str],
    gt_chunk_counter: Counter[str],
) -> float:
    total_eval = max(rows_total, 1)
    coverage = 1.0 - (queries_with_no_gt / total_eval)
    source_diversity_penalty = top_share(gt_url_counter, 1)
    gt_concentration_penalty = top_share(gt_chunk_counter, 10)
    return max(0.0, min(1.0, coverage * (1.0 - 0.5 * source_diversity_penalty) * (1.0 - 0.5 * gt_concentration_penalty)))


def audit(rag_path: Path, eval_path: Path) -> dict[str, Any]:
    raw_chunks = load_raw_chunks(rag_path)
    eval_rows = load_eval_rows(eval_path)

    chunk_stats = _build_chunk_level_stats(raw_chunks)
    eval_stats = _collect_evaluation_counters(eval_rows, chunk_stats["chunk_ids"])
    gt_url_counter = _build_gt_url_counter(eval_stats["gt_chunk_counter"], chunk_stats["chunk_to_url"])

    evidence_dist = Counter(eval_stats["evidence_counts"])
    queries_multi_gt = sum(1 for n in eval_stats["evidence_counts"] if n > 1)
    queries_single_gt = sum(1 for n in eval_stats["evidence_counts"] if n == 1)
    queries_no_gt = sum(1 for n in eval_stats["evidence_counts"] if n == 0)

    quality_score = _quality_score(
        rows_total=len(eval_rows),
        queries_with_no_gt=queries_no_gt,
        gt_url_counter=gt_url_counter,
        gt_chunk_counter=eval_stats["gt_chunk_counter"],
    )

    return {
        "inputs": {"rag_path": str(rag_path), "eval_path": str(eval_path)},
        "rag": {
            "raw_chunks": len(raw_chunks),
            "token_count_min": min(chunk_stats["token_counts"]) if chunk_stats["token_counts"] else 0,
            "token_count_max": max(chunk_stats["token_counts"]) if chunk_stats["token_counts"] else 0,
            "token_count_avg": (
                (sum(chunk_stats["token_counts"]) / len(chunk_stats["token_counts"]))
                if chunk_stats["token_counts"]
                else 0.0
            ),
            "overlap_tokens_unique": sorted(set(chunk_stats["overlap_tokens"])),
            "duplicate_chunk_text_entries": chunk_stats["duplicate_text_entries"],
            "top_categories": chunk_stats["categories"].most_common(10),
            "top_urls": chunk_stats["urls"].most_common(10),
        },
        "evaluation": {
            "rows_total": len(eval_rows),
            "evidence_count_distribution": dict(evidence_dist),
            "queries_with_multi_gt": queries_multi_gt,
            "queries_with_single_gt": queries_single_gt,
            "queries_with_no_gt": queries_no_gt,
            "resolution_method_distribution": dict(eval_stats["resolution"]),
            "unknown_chunk_refs": eval_stats["unknown_chunk_refs"],
            "top_gt_chunks": eval_stats["gt_chunk_counter"].most_common(10),
            "top_gt_urls": gt_url_counter.most_common(10),
            "top1_gt_chunk_share": top_share(eval_stats["gt_chunk_counter"], 1),
            "top10_gt_chunk_share": top_share(eval_stats["gt_chunk_counter"], 10),
            "top1_gt_url_share": top_share(gt_url_counter, 1),
        },
        "quality_score": quality_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RAG dataset and evaluation quality.")
    parser.add_argument("--rag", type=Path, default=Path("data/rag_dataset.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/evaluation_with_evidence.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON path to save report.")
    args = parser.parse_args()

    report = audit(args.rag, args.eval)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
