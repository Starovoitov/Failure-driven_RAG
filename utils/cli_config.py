from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_command_name(command: str | None) -> str:
    return (command or "").replace("-", "_")


def collect_option_dest_map(parser: argparse.ArgumentParser) -> dict[str, str]:
    option_dest_map: dict[str, str] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                option_dest_map.update(collect_option_dest_map(subparser))
            continue
        for option in action.option_strings:
            option_dest_map[option] = action.dest
    return option_dest_map


def extract_explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_dest_map = collect_option_dest_map(parser)
    explicit: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            dest = option_dest_map.get(option)
            if dest:
                explicit.add(dest)
            continue
        if token.startswith("-") and token != "-":
            if token in option_dest_map:
                explicit.add(option_dest_map[token])
    return explicit


def load_cli_defaults(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must be a JSON object.")
    commands = payload.get("commands", payload)
    if not isinstance(commands, dict):
        raise ValueError(f"Config at {config_path} must contain object 'commands'.")
    normalized: dict[str, dict[str, Any]] = {}
    for command, command_params in commands.items():
        if not isinstance(command, str) or not isinstance(command_params, dict):
            continue
        normalized[normalize_command_name(command)] = command_params
    return normalized


def load_script_defaults(config_path: Path, script_key: str) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    scripts = payload.get("scripts", {})
    if not isinstance(scripts, dict):
        return {}
    defaults = scripts.get(script_key, {})
    return defaults if isinstance(defaults, dict) else {}


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str],
    config_defaults: dict[str, dict[str, Any]],
) -> None:
    command_name = normalize_command_name(getattr(args, "command", None))
    command_defaults = config_defaults.get(command_name, {})
    if not command_defaults:
        return
    explicit_dests = extract_explicit_cli_dests(parser, argv)
    for dest, value in command_defaults.items():
        if dest in explicit_dests:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)


def validate_required_command_params(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    required_command_params: dict[str, tuple[str, ...]],
) -> None:
    command_name = normalize_command_name(getattr(args, "command", None))
    for param in required_command_params.get(command_name, ()):
        value = getattr(args, param, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            parser.error(f"missing required parameter '{param}' for command '{command_name}'")
