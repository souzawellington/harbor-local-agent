"""Minimal client for Ollama's chat API with tool calling.

The request shape follows the documented `/api/chat` endpoint: a messages
array, a tools array, and `stream: false`. Tool results are sent back as
messages with `role` of `tool` and a `tool_name`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class OllamaError(Exception):
    """The Ollama server could not complete a chat request."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelMessage:
    content: str
    tool_calls: list[ToolCall]
    thinking: str = ""
    raw_tool_calls: list[dict[str, Any]] | None = None


class ChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelMessage: ...


class OllamaClient:
    def __init__(self, host: str, timeout: float, http: httpx.Client | None = None) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._http = http

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelMessage:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0},
        }
        url = f"{self.host}/api/chat"
        try:
            response = self._post(url, payload)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. "
                "Start it with `ollama serve` and pull a model with "
                f"`ollama pull {model}`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama did not respond within {self.timeout:.0f}s at {self.host}."
            ) from exc
        return _parse_response(response, model)

    def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        if self._http is not None:
            return self._http.post(url, json=payload)
        with httpx.Client(timeout=self.timeout) as http:
            return http.post(url, json=payload)


def _parse_response(response: httpx.Response, model: str) -> ModelMessage:
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:400]
        if response.status_code == 404:
            raise OllamaError(
                f"Ollama returned 404 for model {model!r}. "
                f"Pull it with `ollama pull {model}`. {detail}".strip()
            )
        raise OllamaError(f"Ollama error {response.status_code}: {detail}")
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned a response that was not JSON.") from exc
    if not isinstance(data, dict):
        raise OllamaError("Ollama returned a response that was not a JSON object.")
    if "message" not in data and data.get("error"):
        raise OllamaError(str(data["error"]))
    message = data.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama response did not include a message.")
    raw_calls = message.get("tool_calls")
    return ModelMessage(
        content=str(message.get("content") or ""),
        thinking=str(message.get("thinking") or ""),
        tool_calls=_parse_tool_calls(raw_calls),
        raw_tool_calls=raw_calls if isinstance(raw_calls, list) else None,
    )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise OllamaError("tool_calls was not a list")
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            raise OllamaError("tool call was not an object")
        function = item.get("function")
        if not isinstance(function, dict):
            raise OllamaError("tool call missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OllamaError("tool call missing function name")
        calls.append(
            ToolCall(
                name=name.strip(),
                arguments=_parse_arguments(function.get("arguments")),
            )
        )
    return calls


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OllamaError("tool arguments were not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("tool arguments must be a JSON object")
        return parsed
    raise OllamaError("tool arguments had an unexpected type")
