"""In-memory stand-in for the Ollama client."""

from __future__ import annotations

from typing import Any

from harbor.client import ModelMessage


class ScriptedClient:
    def __init__(self, responses: list[ModelMessage]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelMessage:
        self.requests.append({"model": model, "messages": list(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.pop(0)
