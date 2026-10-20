"""Argument checks shared by tools."""

from __future__ import annotations

from typing import Any

from harbor.sandbox import ToolError


def require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} is required")
    return value.strip()


def optional_int(args: dict[str, Any], key: str, default: int, *, upper: int) -> int:
    if key not in args or args[key] is None:
        return default
    value = _as_int(args[key], key)
    if value < 1 or value > upper:
        raise ToolError(f"{key} must be between 1 and {upper}")
    return value


def require_int(args: dict[str, Any], key: str) -> int:
    if key not in args or args[key] is None:
        raise ToolError(f"{key} is required")
    return _as_int(args[key], key)


def _as_int(value: Any, key: str) -> int:
    if type(value) is bool:
        raise ToolError(f"{key} must be an integer")
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit() and text not in {"", "-"}:
            value = int(text)
    if type(value) is not int:
        raise ToolError(f"{key} must be an integer")
    return value
