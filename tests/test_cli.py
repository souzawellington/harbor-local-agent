from pathlib import Path

import pytest
from typer.testing import CliRunner

from harbor.cli import app
from harbor.client import ModelMessage, OllamaError, ToolCall
from tests.fakes import ScriptedClient

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "harbor 0.1.0" in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.stdout
    assert "run" in result.stdout
    assert "tools" in result.stdout
    assert "web" in result.stdout


def test_tools_lists_the_sandboxed_set(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tools", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    for name in ("read_file", "fetch_url", "calculate", "manage_tasks"):
        assert name in result.stdout


def test_run_prints_the_tool_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = ScriptedClient(
        [
            ModelMessage("", [ToolCall("calculate", {"expression": "2+2"})]),
            ModelMessage("The answer is 4.", []),
        ]
    )
    monkeypatch.setattr("harbor.cli.OllamaClient", lambda *args, **kwargs: fake)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    result = runner.invoke(
        app,
        ["run", "what is 2+2?", "--model", "qwen2.5:3b", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "tool  calculate  ok" in result.stdout
    assert '"expression": "2+2"' in result.stdout
    assert "The answer is 4." in result.stdout
    assert fake.requests[0]["model"] == "qwen2.5:3b"
    assert fake.requests[1]["messages"][-1]["content"] == "4"


def test_run_reports_a_dead_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Down:
        def chat(self, **kwargs: object) -> ModelMessage:
            raise OllamaError("Could not reach Ollama at http://127.0.0.1:11434.")

    monkeypatch.setattr("harbor.cli.OllamaClient", lambda *args, **kwargs: Down())
    result = runner.invoke(app, ["run", "hello", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "Could not reach Ollama" in combined


def test_run_stops_after_max_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Loop:
        def chat(self, **kwargs: object) -> ModelMessage:
            return ModelMessage("", [ToolCall("calculate", {"expression": "1+1"})])

    monkeypatch.setattr("harbor.cli.OllamaClient", lambda *args, **kwargs: Loop())
    result = runner.invoke(
        app,
        ["run", "loop", "--workspace", str(tmp_path), "--max-steps", "2"],
    )
    assert result.exit_code == 2
    assert "Stopped after 2" in result.stdout
    assert "tool  calculate  ok" in result.stdout


def test_chat_keeps_going_until_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = ScriptedClient([ModelMessage("Four.", [])])
    monkeypatch.setattr("harbor.cli.OllamaClient", lambda *args, **kwargs: fake)
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(tmp_path)],
        input="what is 2 + 2?\n/exit\n",
    )
    assert result.exit_code == 0
    assert "tools  (none)" in result.stdout
    assert "Four." in result.stdout
    assert "bye" in result.stdout


def test_empty_task_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "   ", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "empty" in (result.stdout + (result.stderr or ""))


def test_bad_setting_exits_before_a_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "nope")
    result = runner.invoke(app, ["tools", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "AGENT_MAX_STEPS" in (result.stdout + (result.stderr or ""))
