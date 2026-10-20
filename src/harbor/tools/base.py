"""Tool definition passed to Ollama and invoked by the agent loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from harbor.sandbox import ToolError


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(arguments, dict):
            return False, "error: arguments must be an object"
        try:
            result = self.handler(arguments)
        except ToolError as exc:
            return False, f"error: {exc}"
        except Exception as exc:
            return False, f"error: {exc.__class__.__name__}: {exc}"
        if not isinstance(result, str):
            return False, f"error: tool {self.name} returned a non-string"
        return True, result
