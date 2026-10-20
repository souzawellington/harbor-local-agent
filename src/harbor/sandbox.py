"""Workspace path checks.

File tools may read Markdown and text only, and only after the resolved path
is confirmed to sit inside the workspace root. Symlinks are followed before
that check, so a link pointing outside the workspace is refused.
"""

from __future__ import annotations

from pathlib import Path

ALLOWED_SUFFIXES = {".md", ".markdown", ".txt"}
MAX_FILE_BYTES = 512_000


class ToolError(Exception):
    """A tool refused or failed a call. The message is safe to show the model."""


def workspace_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ToolError(f"workspace is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_workspace_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ToolError("path is required")
    if "\x00" in relative:
        raise ToolError("path is invalid")
    raw = Path(relative)
    if raw.is_absolute() or relative.startswith("~"):
        raise ToolError("path must stay inside the workspace")
    root = root.resolve()
    try:
        candidate = (root / raw).resolve()
    except OSError as exc:
        raise ToolError("path is invalid") from exc
    if not candidate.is_relative_to(root):
        raise ToolError("path is outside the workspace")
    return candidate


def read_text_file(root: Path, relative: str, *, max_chars: int) -> tuple[Path, str, bool]:
    if max_chars < 1:
        raise ToolError("max_chars must be at least 1")
    path = resolve_workspace_file(root, relative)
    if not path.exists():
        raise ToolError(f"file not found: {relative}")
    if not path.is_file():
        raise ToolError(f"not a file: {relative}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ToolError("only .md, .markdown, and .txt files can be read")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ToolError(f"file is too large ({size} bytes)")
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return path, text, truncated
