"""Web API tests. Ollama is never contacted."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harbor.client import ModelMessage, OllamaError, ToolCall
from harbor.config import load_settings
from harbor.web import Probe, create_app
from tests.fakes import ScriptedClient


def _settings(tmp_path: Path):
    return load_settings(workspace=tmp_path, model="llama3.2:3b", host="http://127.0.0.1:11434")


def _up(host: str) -> Probe:
    return Probe(True, [], None)


def test_index_is_the_chat_page(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), client=ScriptedClient([]), probe=_up)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Tool calls" in response.text
    assert 'id="model"' in response.text
    assert "Dark mode" in response.text


def test_status_reports_a_dead_ollama(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path),
        client=ScriptedClient([]),
        probe=lambda host: Probe(False, [], f"Could not reach Ollama at {host}."),
    )
    response = TestClient(app).get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "Could not reach Ollama" in body["error"]
    assert body["configured_model"] == "llama3.2:3b"
    assert "llama3.2:3b" in body["models"]


def test_status_lists_models_from_the_probe(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path),
        client=ScriptedClient([]),
        probe=lambda host: Probe(True, ["qwen2.5:3b", "llama3.2:3b"], None),
    )
    body = TestClient(app).get("/api/status").json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["models"] == ["qwen2.5:3b", "llama3.2:3b"]


def test_chat_returns_the_tool_timeline(tmp_path: Path) -> None:
    fake = ScriptedClient(
        [
            ModelMessage("", [ToolCall("calculate", {"expression": "(12 + 8) / 4"})]),
            ModelMessage("(12 + 8) / 4 is 5.", []),
        ]
    )
    app = create_app(_settings(tmp_path), client=fake, probe=_up)
    response = TestClient(app).post("/api/chat", json={"message": "what is (12 + 8) / 4?"})
    assert response.status_code == 200
    body = response.json()
    assert body["finished"] is True
    assert body["answer"] == "(12 + 8) / 4 is 5."
    assert body["traces"][0]["name"] == "calculate"
    assert body["traces"][0]["arguments"] == {"expression": "(12 + 8) / 4"}
    assert body["traces"][0]["ok"] is True
    assert body["traces"][0]["summary"] == "5"
    assert fake.requests[0]["model"] == "llama3.2:3b"


def test_chat_uses_the_selected_model(tmp_path: Path) -> None:
    fake = ScriptedClient([ModelMessage("ok", [])])
    app = create_app(_settings(tmp_path), client=fake, probe=_up)
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "hello", "model": "qwen2.5:3b"},
    )
    assert response.status_code == 200
    assert fake.requests[0]["model"] == "qwen2.5:3b"
    assert response.json()["model"] == "qwen2.5:3b"


def test_chat_rejects_an_invalid_model_name(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), client=ScriptedClient([]), probe=_up)
    response = TestClient(app).post("/api/chat", json={"message": "hello", "model": "bad name"})
    assert response.status_code == 400


def test_chat_reports_ollama_down(tmp_path: Path) -> None:
    class Down:
        def chat(self, **kwargs: object) -> ModelMessage:
            raise OllamaError("Could not reach Ollama at http://127.0.0.1:11434.")

    app = create_app(
        _settings(tmp_path),
        client=Down(),
        probe=lambda host: Probe(False, [], "down"),
    )
    response = TestClient(app).post("/api/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "Could not reach Ollama" in response.json()["detail"]


def test_empty_message_is_rejected(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), client=ScriptedClient([]), probe=_up)
    response = TestClient(app).post("/api/chat", json={"message": "   "})
    assert response.status_code == 400


def test_web_chat_still_refuses_paths_outside_the_workspace(tmp_path: Path) -> None:
    fake = ScriptedClient(
        [
            ModelMessage("", [ToolCall("read_file", {"path": "../secrets.txt", "mode": "full"})]),
            ModelMessage("I could not read that file.", []),
        ]
    )
    app = create_app(_settings(tmp_path), client=fake, probe=_up)
    body = TestClient(app).post("/api/chat", json={"message": "read the secret"}).json()
    assert body["traces"][0]["ok"] is False
    assert "outside the workspace" in body["traces"][0]["result"]


def test_reset_clears_the_conversation(tmp_path: Path) -> None:
    fake = ScriptedClient([ModelMessage("one", []), ModelMessage("two", [])])
    app = create_app(_settings(tmp_path), client=fake, probe=_up)
    client = TestClient(app)
    client.post("/api/chat", json={"message": "first"})
    client.post("/api/reset")
    client.post("/api/chat", json={"message": "second"})
    roles = [message["role"] for message in fake.requests[1]["messages"]]
    assert roles == ["system", "user"]


def test_web_help_does_not_start_a_server() -> None:
    from typer.testing import CliRunner

    from harbor.cli import app

    result = CliRunner().invoke(app, ["web", "--help"])
    assert result.exit_code == 0
    assert "--bind" in result.stdout
    assert "--port" in result.stdout
    assert "127.0.0.1" in result.stdout


def test_probe_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from harbor.web import probe_ollama

    def boom(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.Client, "get", boom)
    result = probe_ollama("http://127.0.0.1:11434")
    assert result.ok is False
    assert "Could not reach Ollama" in (result.error or "")
