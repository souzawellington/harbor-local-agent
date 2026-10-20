"""Local web UI. Same agent loop and tools as the CLI. No login."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from harbor.agent import Agent
from harbor.client import ChatClient, OllamaClient, OllamaError
from harbor.config import Settings

_PAGE = Path(__file__).with_name("static") / "index.html"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.:/+-]{1,80}$")


@dataclass(frozen=True)
class Probe:
    ok: bool
    models: list[str]
    error: str | None


class ChatRequest(BaseModel):
    message: str = Field(default="")
    model: str | None = None


def probe_ollama(host: str, timeout: float = 2.5) -> Probe:
    """Ask Ollama for its model list. A connection failure is an error state, not a crash."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=timeout) as http:
            response = http.get(url)
    except httpx.ConnectError:
        return Probe(
            ok=False,
            models=[],
            error=(
                f"Could not reach Ollama at {host}. "
                "Start it with `ollama serve`, then pull a model with `ollama pull llama3.2:3b`."
            ),
        )
    except httpx.TimeoutException:
        return Probe(
            ok=False,
            models=[],
            error=f"Ollama did not respond at {host}.",
        )
    if response.status_code >= 400:
        return Probe(
            ok=False,
            models=[],
            error=f"Ollama returned {response.status_code} from /api/tags.",
        )
    try:
        data = response.json()
    except ValueError:
        return Probe(ok=False, models=[], error="Ollama /api/tags was not JSON.")
    models: list[str] = []
    if isinstance(data, dict):
        raw = data.get("models")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    name = item["name"].strip()
                    if name and name not in models:
                        models.append(name)
    return Probe(ok=True, models=models, error=None)


def create_app(
    settings: Settings,
    *,
    client: ChatClient | None = None,
    probe: Callable[[str], Probe] | None = None,
) -> FastAPI:
    """Build the UI app. Tests pass a scripted client and probe so Ollama stays mocked."""
    chat_client = client or OllamaClient(settings.ollama_host, settings.timeout_seconds)
    agent = Agent(
        client=chat_client,
        model=settings.model,
        tools=_tools(settings),
        max_steps=settings.max_steps,
    )
    lock = threading.Lock()
    checker = probe or (lambda host: probe_ollama(host))
    app = FastAPI(title="Harbor", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.agent = agent
    app.state.lock = lock
    app.state.probe = checker

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        result = app.state.probe(settings.ollama_host)
        models = list(result.models)
        if settings.model not in models:
            models.insert(0, settings.model)
        return {
            "ok": result.ok,
            "model": app.state.agent.model,
            "configured_model": settings.model,
            "host": settings.ollama_host,
            "workspace": str(settings.workspace),
            "models": models,
            "error": result.error,
        }

    @app.post("/api/chat")
    def chat(body: ChatRequest) -> dict[str, Any]:
        text = body.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="message is empty")
        chosen_model = body.model.strip() if body.model else ""
        if chosen_model and not _MODEL_NAME.fullmatch(chosen_model):
            raise HTTPException(status_code=400, detail="model name is invalid")
        with app.state.lock:
            if chosen_model:
                app.state.agent.model = chosen_model
            try:
                result = app.state.agent.ask(text)
            except OllamaError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "answer": result.answer,
            "finished": result.finished,
            "model": app.state.agent.model,
            "traces": [
                {
                    "name": trace.name,
                    "arguments": trace.arguments,
                    "result": trace.result,
                    "ok": trace.ok,
                    "summary": _summary(trace.result),
                }
                for trace in result.traces
            ],
        }

    @app.post("/api/reset")
    def reset() -> dict[str, bool]:
        with app.state.lock:
            system = app.state.agent.messages[:1]
            app.state.agent.messages = system
        return {"ok": True}

    return app


def _tools(settings: Settings) -> list[Any]:
    from harbor.tools import build_registry

    return build_registry(settings.workspace)


def _summary(result: str, limit: int = 280) -> str:
    text = " ".join(result.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
