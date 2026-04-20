#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def cmd_build_parser(args: argparse.Namespace) -> None:
    from parser.pipeline import run_pipeline

    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
    )
    print(json.dumps(stats, indent=2))


def cmd_demo_retrieval(args: argparse.Namespace) -> None:
    from demo_retrieval import run_demo

    run_demo(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        dataset_path=args.dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
    )


def cmd_evaluation_runner(args: argparse.Namespace) -> None:
    from pathlib import Path

    from evaluation.dataset import load_eval_samples
    from evaluation.metrics import RetrievalResult, evaluate_retrieval
    from evaluation.runner import QueryRun, build_retriever, parse_k_values
    from ingestion.loaders import load_bm25_documents_from_dataset

    samples = load_eval_samples(Path(args.dataset))
    if not samples:
        raise ValueError(f"No samples found in dataset: {args.dataset}")

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)
    retriever = build_retriever(
        args.retriever,
        rag_dataset_path=args.rag_dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
    )
    doc_text_map = {
        item["id"]: item["text"] for item in load_bm25_documents_from_dataset(args.rag_dataset)
    }
    reranker = None
    if args.rerank:
        from reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=args.reranker_model)

    query_runs: list[QueryRun] = []
    metric_inputs: list[RetrievalResult] = []
    for sample in samples:
        retrieve_k = max(max_k, args.rerank_candidates) if args.rerank else max_k
        retrieved = retriever.search(sample.query, top_k=retrieve_k)
        if reranker is not None:
            from reranking.cross_encoder import RerankCandidate

            rerank_input = [
                RerankCandidate(
                    doc_id=doc_id,
                    text=doc_text_map.get(doc_id, ""),
                )
                for doc_id in retrieved
                if doc_text_map.get(doc_id, "")
            ]
            reranked = reranker.rerank(sample.query, rerank_input, top_k=max_k)
            retrieved = [item.doc_id for item in reranked]
        else:
            retrieved = retrieved[:max_k]
        query_runs.append(
            QueryRun(query=sample.query, relevant_doc_ids=sample.relevant_docs, retrieved_doc_ids=retrieved)
        )
        metric_inputs.append(
            RetrievalResult(
                query=sample.query,
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=sample.relevant_docs,
            )
        )

    metrics = evaluate_retrieval(metric_inputs, k_values)
    report = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rerank_enabled": args.rerank,
        "reranker_model": args.reranker_model if args.rerank else None,
        "k_values": k_values,
        "samples_total": len(samples),
        "samples_with_ground_truth": sum(1 for s in samples if s.relevant_docs),
        "metrics": metrics,
        "runs": [run.__dict__ for run in query_runs],
    }

    print("Retrieval benchmark report")
    print(f"- dataset: {args.dataset}")
    print(f"- retriever: {args.retriever}")
    print(f"- samples: {len(samples)}")
    for key in sorted(metrics):
        print(f"- {key}: {metrics[key]:.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")


def cmd_run_rag(args: argparse.Namespace) -> None:
    from generation.run_rag import run_rag

    run_rag(
        question=args.question,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
    )


def cmd_cleanup_faiss(args: argparse.Namespace) -> None:
    from ingestion.cleaner import cleanup_faiss_db

    result = cleanup_faiss_db(
        persist_directory=args.faiss_path,
        index_name=args.index,
        drop_persist_directory=args.drop_persist_directory,
    )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entrypoint for project workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser(
        "build_parser",
        help="Run parser pipeline and build rag_dataset.jsonl",
    )
    build_parser_cmd.add_argument("--output", default="data/rag_dataset.jsonl")
    build_parser_cmd.add_argument("--min-tokens", type=int, default=300)
    build_parser_cmd.add_argument("--max-tokens", type=int, default=800)
    build_parser_cmd.add_argument("--overlap-ratio", type=float, default=0.15)
    build_parser_cmd.set_defaults(handler=cmd_build_parser)

    demo_cmd = subparsers.add_parser("demo_retrieval", help="Run BM25/semantic/hybrid retrieval demo.")
    demo_cmd.add_argument("--query", "-q", default="database caching performance")
    demo_cmd.add_argument("--top-k", "-k", type=int, default=4)
    demo_cmd.add_argument("--model", "-m", default="intfloat/e5-small-v2")
    demo_cmd.add_argument("--dataset", default="data/rag_dataset.jsonl")
    demo_cmd.add_argument("--faiss-path", default="data/faiss")
    demo_cmd.add_argument("--index", default="rag_chunks")
    demo_cmd.add_argument("--rerank", action="store_true")
    demo_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    demo_cmd.add_argument("--rerank-candidates", type=int, default=20)
    demo_cmd.set_defaults(handler=cmd_demo_retrieval)

    eval_cmd = subparsers.add_parser("evaluation_runner", help="Run retrieval benchmark over eval dataset.")
    eval_cmd.add_argument("--dataset", default="data/evaluation_with_evidence.jsonl")
    eval_cmd.add_argument("--retriever", choices=("semantic", "bm25", "hybrid"), default="semantic")
    eval_cmd.add_argument("--k-values", default="1,3,5")
    eval_cmd.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    eval_cmd.add_argument("--faiss-path", default="data/faiss")
    eval_cmd.add_argument("--index", default="rag_chunks")
    eval_cmd.add_argument("--embedding-model", default="intfloat/e5-small-v2")
    eval_cmd.add_argument("--alpha", type=float, default=0.7)
    eval_cmd.add_argument("--rerank", action="store_true")
    eval_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    eval_cmd.add_argument("--rerank-candidates", type=int, default=20)
    eval_cmd.add_argument("--out-json", default=None)
    eval_cmd.set_defaults(handler=cmd_evaluation_runner)

    rag_cmd = subparsers.add_parser("run_rag", help="Run full RAG query against selected LLM provider.")
    rag_cmd.add_argument("--question", "-q", required=True)
    rag_cmd.add_argument("--provider", default="openai", choices=("openai", "gigachat", "ollama", "qwen"))
    rag_cmd.add_argument("--model", default=None)
    rag_cmd.add_argument("--top-k", type=int, default=5)
    rag_cmd.add_argument("--max-context-tokens", type=int, default=2500)
    rag_cmd.add_argument("--faiss-path", default="data/faiss")
    rag_cmd.add_argument("--index", default="rag_chunks")
    rag_cmd.add_argument("--embedding-model", default="intfloat/e5-small-v2")
    rag_cmd.add_argument("--stream", action="store_true")
    rag_cmd.add_argument("--max-tokens", type=int, default=512)
    rag_cmd.add_argument("--temperature", type=float, default=0.1)
    rag_cmd.add_argument("--top-p", type=float, default=0.95)
    rag_cmd.add_argument("--rerank", action="store_true")
    rag_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rag_cmd.add_argument("--rerank-candidates", type=int, default=20)
    rag_cmd.set_defaults(handler=cmd_run_rag)

    clean_cmd = subparsers.add_parser("cleanup_faiss", help="Delete FAISS index and optionally full directory.")
    clean_cmd.add_argument("--faiss-path", default="data/faiss")
    clean_cmd.add_argument("--index", default="rag_chunks")
    clean_cmd.add_argument("--drop-persist-directory", action="store_true")
    clean_cmd.set_defaults(handler=cmd_cleanup_faiss)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
