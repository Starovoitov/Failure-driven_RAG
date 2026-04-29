from __future__ import annotations

import argparse
import contextlib
import io
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model

from main import REQUIRED_COMMAND_PARAMS, build_parser
from utils.cli_config import apply_config_defaults, load_cli_defaults, validate_required_command_params


DEFAULT_CLI_PARAMS_CONFIG = "cli.defaults.json"
SUPPORTED_COMMANDS = (
    "build_parser",
    "build_faiss",
    "demo_retrieval",
    "evaluation_runner",
    "reranker_pipeline",
    "run_rag",
    "cleanup_faiss",
)
CLI_DEFAULTS = load_cli_defaults(Path.cwd() / DEFAULT_CLI_PARAMS_CONFIG)


class CommandResponse(BaseModel):
    command: str
    argv: list[str]
    stdout: str
    stderr: str
    result: dict[str, Any] | None = None


class CommandTaskStartResponse(BaseModel):
    task_id: str
    command: str
    argv: list[str]
    status: Literal["running"]


class CommandTaskStatusResponse(BaseModel):
    task_id: str
    command: str
    argv: list[str]
    status: Literal["running", "completed", "failed"]
    stdout: str
    stderr: str
    result: dict[str, Any] | None = None
    error: str | None = None


class FileStatusRequest(BaseModel):
    paths: list[str]


class FileStatusItem(BaseModel):
    path: str
    exists: bool
    is_dir: bool = False
    size_bytes: int | None = None
    modified_ts: float | None = None


class FileStatusResponse(BaseModel):
    items: list[FileStatusItem]


@dataclass
class CommandSpec:
    model: type[BaseModel]
    actions: dict[str, argparse.Action]
    example_payload: dict[str, Any]


@dataclass
class CommandTask:
    task_id: str
    command: str
    argv: list[str]
    status: Literal["running", "completed", "failed"]
    stdout_buffer: io.StringIO
    stderr_buffer: io.StringIO
    result: dict[str, Any] | None = None
    error: str | None = None


TASKS: dict[str, CommandTask] = {}
TASKS_LOCK = threading.Lock()


def _extract_subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("No subparsers found in CLI parser.")


def _action_field_type(action: argparse.Action) -> Any:
    if isinstance(action, argparse._StoreTrueAction):
        return bool

    if action.choices:
        literals = tuple(action.choices)
        return Literal.__getitem__(literals)  # type: ignore[misc]

    value_type = action.type if action.type in {int, float, str} else str
    return value_type | None


def _build_command_spec(
    subparser: argparse.ArgumentParser,
    config_defaults: dict[str, Any],
) -> CommandSpec:
    fields: dict[str, tuple[Any, Any]] = {}
    actions: dict[str, argparse.Action] = {}
    for action in subparser._actions:
        if not action.option_strings:
            continue
        if action.dest in {"help"}:
            continue

        field_type = _action_field_type(action)
        if action.dest in config_defaults:
            schema_default = config_defaults[action.dest]
        elif isinstance(action, argparse._StoreTrueAction):
            schema_default = False
        else:
            schema_default = None
        default = Field(default=schema_default, description=action.help)

        fields[action.dest] = (field_type, default)
        actions[action.dest] = action

    model_name = f"{subparser.prog.replace(' ', '_')}_Request"
    example_payload = {key: value for key, value in config_defaults.items() if key in fields}
    model = create_model(
        model_name,
        __base__=BaseModel,
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={"example": example_payload} if example_payload else None,
        ),
        **fields,
    )
    return CommandSpec(model=model, actions=actions, example_payload=example_payload)


def _build_command_specs() -> dict[str, CommandSpec]:
    parser = build_parser()
    subparsers = _extract_subparsers(parser)
    specs: dict[str, CommandSpec] = {}
    command_defaults = CLI_DEFAULTS
    for command_name in SUPPORTED_COMMANDS:
        subparser = subparsers.choices.get(command_name)
        if subparser is None:
            raise RuntimeError(f"Command parser not found for '{command_name}'.")
        specs[command_name] = _build_command_spec(
            subparser=subparser,
            config_defaults=command_defaults.get(command_name, {}),
        )
    return specs


COMMAND_SPECS = _build_command_specs()

app = FastAPI(
    title="RAG FD Command API",
    description=(
        "REST wrapper over primary CLI commands with full flag coverage.\n\n"
        "Swagger UI: `/docs`\n"
        "OpenAPI schema: `/openapi.json`"
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "commands", "description": "Execute primary CLI workflows via REST."},
        {"name": "meta", "description": "Service health and metadata endpoints."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_argv(command: str, payload: BaseModel, actions: dict[str, argparse.Action]) -> list[str]:
    argv = [command]
    values = payload.model_dump(exclude_unset=True)
    for dest, value in values.items():
        action = actions[dest]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
            continue
        if value is None:
            continue
        argv.extend([action.option_strings[0], str(value)])
    return argv


def execute_cli_command(command: str, payload: BaseModel) -> CommandResponse:
    parser = build_parser()
    spec = COMMAND_SPECS[command]
    argv = _build_argv(command, payload, spec.actions)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    return _execute_with_captured_streams(parser, command, argv, stdout_buffer, stderr_buffer)


def _execute_with_captured_streams(
    parser: argparse.ArgumentParser,
    command: str,
    argv: list[str],
    stdout_buffer: io.StringIO,
    stderr_buffer: io.StringIO,
) -> CommandResponse:
    try:
        args = parser.parse_args(argv)
        config_path = Path.cwd() / DEFAULT_CLI_PARAMS_CONFIG
        config_defaults = load_cli_defaults(config_path)
        apply_config_defaults(parser, args, argv, config_defaults)
        validate_required_command_params(parser, args, REQUIRED_COMMAND_PARAMS)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            args.handler(args)
    except SystemExit as exc:
        raise HTTPException(
            status_code=400,
            detail={"command": command, "argv": argv, "error": f"Argument parsing failed: {exc}"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"command": command, "argv": argv, "error": str(exc)},
        ) from exc

    stdout = stdout_buffer.getvalue()
    stderr = stderr_buffer.getvalue()
    parsed_json: dict[str, Any] | None = None
    if stdout.strip().startswith("{"):
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                parsed_json = parsed
        except json.JSONDecodeError:
            parsed_json = None

    return CommandResponse(
        command=command,
        argv=argv,
        stdout=stdout,
        stderr=stderr,
        result=parsed_json,
    )


def _run_task_worker(task_id: str, command: str, payload: BaseModel) -> None:
    parser = build_parser()
    spec = COMMAND_SPECS[command]
    argv = _build_argv(command, payload, spec.actions)
    with TASKS_LOCK:
        task = TASKS[task_id]
        task.argv = argv
    try:
        response = _execute_with_captured_streams(
            parser=parser,
            command=command,
            argv=argv,
            stdout_buffer=task.stdout_buffer,
            stderr_buffer=task.stderr_buffer,
        )
        with TASKS_LOCK:
            task.result = response.result
            task.status = "completed"
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        with TASKS_LOCK:
            task.error = json.dumps(detail, ensure_ascii=False)
            task.status = "failed"
    except Exception as exc:  # noqa: BLE001
        with TASKS_LOCK:
            task.error = str(exc)
            task.status = "failed"


def start_async_task(command: str, payload: BaseModel) -> CommandTaskStartResponse:
    task_id = uuid.uuid4().hex
    task = CommandTask(
        task_id=task_id,
        command=command,
        argv=[],
        status="running",
        stdout_buffer=io.StringIO(),
        stderr_buffer=io.StringIO(),
    )
    with TASKS_LOCK:
        TASKS[task_id] = task

    thread = threading.Thread(
        target=_run_task_worker,
        args=(task_id, command, payload),
        daemon=True,
    )
    thread.start()
    return CommandTaskStartResponse(task_id=task_id, command=command, argv=task.argv, status="running")


def get_task_status(task_id: str) -> CommandTaskStatusResponse:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail={"error": f"Task not found: {task_id}"})
        return CommandTaskStatusResponse(
            task_id=task.task_id,
            command=task.command,
            argv=task.argv,
            status=task.status,
            stdout=task.stdout_buffer.getvalue(),
            stderr=task.stderr_buffer.getvalue(),
            result=task.result,
            error=task.error,
        )


def _register_command_route(command: str, spec: CommandSpec) -> None:
    body = (
        Body(..., examples={"default": {"summary": "Config defaults", "value": spec.example_payload}})
        if spec.example_payload
        else Body(...)
    )

    def _run(payload: BaseModel = body) -> CommandResponse:
        return execute_cli_command(command, payload)

    _run.__annotations__ = {"payload": spec.model, "return": CommandResponse}
    app.post(
        f"/{command}",
        response_model=CommandResponse,
        tags=["commands"],
        operation_id=f"run_{command}",
        summary=f"Run `{command}` command",
        description=(
            f"Executes the `{command}` CLI command. "
            "Request body fields map directly to command flags."
        ),
    )(_run)

    def _run_async(payload: BaseModel = body) -> CommandTaskStartResponse:
        return start_async_task(command, payload)

    _run_async.__annotations__ = {"payload": spec.model, "return": CommandTaskStartResponse}
    app.post(
        f"/{command}/async",
        response_model=CommandTaskStartResponse,
        tags=["commands"],
        operation_id=f"run_{command}_async",
        summary=f"Run `{command}` command asynchronously",
        description=(
            f"Starts `{command}` in background and returns task id for polling "
            "via `GET /tasks/{task_id}`."
        ),
    )(_run_async)


for command_name, command_spec in COMMAND_SPECS.items():
    _register_command_route(command_name, command_spec)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks/{task_id}", response_model=CommandTaskStatusResponse, tags=["meta"])
def task_status(task_id: str) -> CommandTaskStatusResponse:
    return get_task_status(task_id)


@app.post("/files/status", response_model=FileStatusResponse, tags=["meta"])
def files_status(payload: FileStatusRequest) -> FileStatusResponse:
    items: list[FileStatusItem] = []
    cwd = Path.cwd()
    for raw_path in payload.paths:
        resolved = (cwd / raw_path).resolve()
        try:
            relative = resolved.relative_to(cwd.resolve())
        except ValueError:
            # Skip paths outside workspace root.
            items.append(FileStatusItem(path=raw_path, exists=False))
            continue

        path = cwd / relative
        if not path.exists():
            items.append(FileStatusItem(path=raw_path, exists=False))
            continue

        stat = path.stat()
        items.append(
            FileStatusItem(
                path=raw_path,
                exists=True,
                is_dir=path.is_dir(),
                size_bytes=None if path.is_dir() else stat.st_size,
                modified_ts=stat.st_mtime,
            )
        )
    return FileStatusResponse(items=items)

