from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from parser.models import SourceSpec

DEFAULT_SOURCES_CONFIG_PATH = "sources.config.json"


class SourceSpecPayload(BaseModel):
    category: str
    subtopic: str
    url: str
    source_type: str
    priority_topics: list[str]


class SourcesConfigPayload(BaseModel):
    sources: list[SourceSpecPayload]


class AliasGroupPayload(BaseModel):
    primary: str
    aliases: list[str]


class SeedChunkPayload(BaseModel):
    title: str
    content: str


def build_sources(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> list[SourceSpec]:
    """Load source specs from a JSON config file."""
    raw = _load_sources_config(config_path=config_path)
    try:
        payload = SourcesConfigPayload.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid sources config '{config_path}': {exc}") from exc
    return [_parse_source_spec(item, idx) for idx, item in enumerate(payload.sources)]


def build_alias_groups(
    config_path: str = DEFAULT_SOURCES_CONFIG_PATH,
) -> tuple[AliasGroupPayload, ...]:
    """Load alias groups used for chunk enrichment."""
    raw = _load_sources_config(config_path=config_path)
    payload = raw.get("alias_groups", [])
    if not isinstance(payload, list):
        raise ValueError(f"Invalid sources config '{config_path}': 'alias_groups' must be a list")
    try:
        parsed = TypeAdapter(list[AliasGroupPayload]).validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid sources config '{config_path}': {exc}") from exc

    result: list[AliasGroupPayload] = []
    for idx, item in enumerate(parsed):
        primary = str(item.primary).strip()
        aliases = item.aliases
        if not primary:
            raise ValueError(f"alias_groups[{idx}].primary must be a non-empty string")
        cleaned_aliases = [alias.strip() for alias in aliases if alias.strip()]
        result.append(AliasGroupPayload(primary=primary, aliases=cleaned_aliases))
    return tuple(result)


def build_seed_chunks(
    config_path: str = DEFAULT_SOURCES_CONFIG_PATH,
) -> tuple[SeedChunkPayload, ...]:
    """Load synthetic seed chunks that are always appended to dataset output."""
    raw = _load_sources_config(config_path=config_path)
    payload = raw.get("multi_hop_seed_chunks", [])
    if not isinstance(payload, list):
        raise ValueError(
            f"Invalid sources config '{config_path}': 'multi_hop_seed_chunks' must be a list"
        )
    try:
        parsed = TypeAdapter(list[SeedChunkPayload]).validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid sources config '{config_path}': {exc}") from exc

    result: list[SeedChunkPayload] = []
    for idx, item in enumerate(parsed):
        title = str(item.title).strip()
        content = str(item.content).strip()
        if not title:
            raise ValueError(f"multi_hop_seed_chunks[{idx}].title must be a non-empty string")
        if not content:
            raise ValueError(f"multi_hop_seed_chunks[{idx}].content must be a non-empty string")
        result.append(SeedChunkPayload(title=title, content=content))
    return tuple(result)


def _load_sources_config(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sources config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid sources config '{path}': expected top-level JSON object")
    return raw


def _parse_source_spec(payload: SourceSpecPayload, idx: int) -> SourceSpec:
    priority_topics = payload.priority_topics
    return SourceSpec(
        category=str(payload.category).strip(),
        subtopic=str(payload.subtopic).strip(),
        url=str(payload.url).strip(),
        source_type=str(payload.source_type).strip(),
        priority_topics=[item.strip() for item in priority_topics if str(item).strip()],
    )
