from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parser.pipeline import run_pipeline
from utils.cli_config import load_script_defaults


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for running the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Parse RAG sources into JSONL dataset (chunks, Q/A, edge cases)."
    )
    parser.add_argument("--config", help="Path to CLI defaults JSON.")
    parser.add_argument(
        "--output",

        help="Output JSONL path.",
    )
    parser.add_argument("--min-tokens", type=int,)
    parser.add_argument("--max-tokens", type=int,)
    parser.add_argument("--overlap-ratio", type=float,)
    parser.add_argument("--min-output-chunk-tokens", type=int,)
    parser.add_argument("--max-output-chunk-tokens", type=int,)
    parser.add_argument("--max-chunks-per-url", type=int,)
    parser.add_argument("--max-chunks-per-category", type=int,)
    parser.add_argument("--sources-config",)
    parser.add_argument("--chunker-mode", choices=("token", "semantic_dynamic"),)
    parser.add_argument("--near-duplicate-jaccard", type=float,)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-path",)
    parser.add_argument("--log-json", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments, run pipeline, and print run statistics."""
    parser = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _ = pre_parser.parse_known_args(sys.argv[1:])
    config_path = Path(pre_args.config).expanduser() if pre_args.config else (Path.cwd() / "cli.defaults.json")
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    parser.set_defaults(**load_script_defaults(config_path, "parser_main"))
    args = parser.parse_args()
    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
        min_output_chunk_tokens=args.min_output_chunk_tokens,
        max_output_chunk_tokens=args.max_output_chunk_tokens,
        max_chunks_per_url=args.max_chunks_per_url,
        max_chunks_per_category=args.max_chunks_per_category,
        sources_config=args.sources_config,
        chunker_mode=args.chunker_mode,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        log_level=args.log_level,
        log_path=args.log_path,
        log_json=args.log_json,
    )
    print(json.dumps(stats, indent=2))


