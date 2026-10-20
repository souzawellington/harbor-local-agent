"""Settings from the environment and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from harbor.sandbox import ToolError, workspace_root

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_WORKSPACE = "workspace"
DEFAULT_MAX_STEPS = 6
DEFAULT_TIMEOUT = 120.0


class ConfigError(Exception):
    """The process was started with invalid settings."""


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    model: str
    workspace: Path
    max_steps: int
    timeout_seconds: float


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding real variables."""
    env_path = path if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_settings(
    *,
    model: str | None = None,
    workspace: Path | None = None,
    host: str | None = None,
    max_steps: int | None = None,
    env_file: Path | None = None,
) -> Settings:
    load_dotenv(env_file)
    chosen_host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip()
    chosen_model = (model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL).strip()
    if not chosen_host:
        raise ConfigError("OLLAMA_HOST is empty")
    if not chosen_model:
        raise ConfigError("OLLAMA_MODEL is empty")
    workspace_value = (
        workspace
        if workspace is not None
        else Path(os.environ.get("AGENT_WORKSPACE") or DEFAULT_WORKSPACE)
    )
    try:
        root = workspace_root(workspace_value)
    except ToolError as exc:
        raise ConfigError(str(exc)) from exc
    return Settings(
        ollama_host=chosen_host.rstrip("/"),
        model=chosen_model,
        workspace=root,
        max_steps=max_steps if max_steps is not None else _int_env("AGENT_MAX_STEPS"),
        timeout_seconds=_float_env("OLLAMA_TIMEOUT"),
    )


def _int_env(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_STEPS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < 1 or value > 20:
        raise ConfigError(f"{name} must be between 1 and 20")
    return value


def _float_env(name: str) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if value < 1 or value > 600:
        raise ConfigError(f"{name} must be between 1 and 600")
    return value
