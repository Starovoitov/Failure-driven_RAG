from __future__ import annotations

import json
import os
from pathlib import Path
from pydantic import BaseModel, ValidationError

from generation.llm import LLMConfig

DEFAULT_LLM_CONFIG_PATH = "llm.config.json"


class LLMProviderPayload(BaseModel):
    provider: str
    model: str
    api_base: str
    api_key_env: str


class LLMProvidersConfigPayload(BaseModel):
    providers: dict[str, LLMProviderPayload]


def load_llm_provider_configs(config_path: str = DEFAULT_LLM_CONFIG_PATH) -> dict[str, LLMConfig]:
    """Load named LLM provider defaults from JSON config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"LLM config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        cfg = LLMProvidersConfigPayload.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid llm config '{path}': {exc}") from exc
    providers = cfg.providers
    if not providers:
        raise ValueError(f"Invalid llm config '{path}': top-level 'providers' must be a non-empty object")

    result: dict[str, LLMConfig] = {}
    for name, payload in providers.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Provider names must be non-empty strings")
        env_name = payload.api_key_env
        result[name] = LLMConfig(
            provider=payload.provider.strip(),
            model=payload.model.strip(),
            api_base=payload.api_base.strip(),
            api_key=os.getenv(env_name),
        )
    return result
