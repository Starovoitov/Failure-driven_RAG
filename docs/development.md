# Development tooling

This project uses `black` for formatting and `ruff` for linting.

## Install dependencies

From the repository root:

```bash
make install
```

This installs runtime and development dependencies via Poetry.

## Run linting checks

```bash
make lint
```

The lint target runs:

- `black --check .`
- `ruff check .`

## Apply formatter and linter autofixes

```bash
make fix
```

The fix target runs:

- `black .`
- `ruff check --fix .`

## Optional direct commands

If you prefer calling tools directly:

```bash
poetry run black .
poetry run ruff check --fix .
```

## Parser replacement contract (input/output reference)

If you plan to replace the `parser` module, keep the same data contract so downstream
commands (`build_faiss`, `build_evaluation_dataset`, `evaluation_runner`, `run_rag`) continue
to work.

### Input contract

- Source config file is loaded by `parser.sources.build_sources()` from `sources.config.json`
  (or `--sources-config`).
- Top-level source payload model: `parser.sources.SourcesConfigPayload`
  with `sources: list[SourceSpecPayload]`.
- Per-source model: `parser.sources.SourceSpecPayload` with fields:
  - `category: str`
  - `subtopic: str`
  - `url: str`
  - `source_type: str`
  - `priority_topics: list[str]`
- Runtime source object used by pipeline stages: `parser.models.SourceSpec`.
- Optional enrichment inputs from the same config:
  - `parser.sources.AliasGroupPayload` (`alias_groups`)
  - `parser.sources.SeedChunkPayload` (`multi_hop_seed_chunks`)

### Output contract (JSONL records)

The parser output file (default: `data/rag_dataset.jsonl`) is newline-delimited JSON with
mixed record types.

Pydantic-backed record models (defined in `parser.models`):

- `RawChunkRecord` (`record_type="raw_chunk"`)
  - Core fields: `chunk_id`, `text`, `token_count`, `overlap_tokens`, `metadata`
  - `metadata` is built in `parser.pipeline.enrich_metadata()` and includes:
    `chunk_id`, `url`, `title`, `category`, `subtopic`, `source_type`,
    `priority_topics`, `chunk_index`, `language`, `scraped_at`
- `QAPairRecord` (`record_type="qa_pair"`) for synthetic QA supervision
- `EdgeCaseRecord` (`record_type="edge_case"`) for failure-pattern examples

Additional non-model record:

- `source_error` records are currently written as plain dicts in `parser.pipeline.run_pipeline()`
  when scraping fails.

### Pipeline return value contract

- `parser.pipeline.run_pipeline()` returns `PipelineStats.model_dump()`.
- Stats model: `parser.pipeline.PipelineStats` with counters such as:
  `raw_chunks`, `qa_pairs`, `edge_cases`, `sources_ok`, and skip counters.

### Compatibility checklist for a new parser implementation

- Keep `raw_chunk` JSON structure compatible with `RawChunkRecord`.
- Preserve `metadata.chunk_id` generation semantics (stable per URL/chunk content) or provide
  a deterministic equivalent to avoid retrieval/evaluation drift.
- Keep `qa_pair` and `edge_case` record shapes compatible with `QAPairRecord` and `EdgeCaseRecord`.
- Keep `run_pipeline()` return keys aligned with `PipelineStats`.
