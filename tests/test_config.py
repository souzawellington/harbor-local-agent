from pathlib import Path

import pytest

from harbor.config import ConfigError, load_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
        "AGENT_WORKSPACE",
        "AGENT_MAX_STEPS",
        "OLLAMA_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(workspace=tmp_path, env_file=tmp_path / "missing.env")
    assert settings.model == "llama3.2:3b"
    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.workspace == tmp_path.resolve()
    assert settings.max_steps == 6
    assert settings.timeout_seconds == 120


def test_dotenv_does_not_override_existing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-file\n", encoding="utf-8")
    settings = load_settings(workspace=tmp_path / "ws", env_file=env_file)
    assert settings.model == "from-env"


def test_dotenv_fills_missing_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        'export OLLAMA_MODEL="qwen2.5:3b"\nOLLAMA_HOST=http://127.0.0.1:11435/\n',
        encoding="utf-8",
    )
    settings = load_settings(workspace=tmp_path / "ws", env_file=env_file)
    assert settings.model == "qwen2.5:3b"
    assert settings.ollama_host == "http://127.0.0.1:11435"


def test_explicit_arguments_win(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    settings = load_settings(
        model="from-flag",
        host="http://localhost:11434/",
        max_steps=3,
        workspace=tmp_path,
        env_file=tmp_path / "missing.env",
    )
    assert settings.model == "from-flag"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.max_steps == 3


def test_invalid_max_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "99")
    with pytest.raises(ConfigError, match="between 1 and 20"):
        load_settings(workspace=tmp_path, env_file=tmp_path / "missing.env")


def test_workspace_must_be_a_directory(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        load_settings(workspace=blocked, env_file=tmp_path / "missing.env")
