"""Read a Markdown or text file inside the workspace."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harbor.sandbox import ToolError, read_text_file
from harbor.tools.base import Tool
from harbor.tools.common import optional_int, require_str

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_OUTLINE_MODES = {"outline", "summary", "summarise", "summarize"}


def make_read_file(root: Path) -> Tool:
    def handler(args: dict[str, Any]) -> str:
        relative = require_str(args, "path")
        mode = args.get("mode", "full")
        if mode is None:
            mode = "full"
        if not isinstance(mode, str):
            raise ToolError("mode must be a string")
        mode = mode.strip().lower()
        if mode in _OUTLINE_MODES:
            mode = "outline"
        if mode not in {"full", "outline"}:
            raise ToolError("mode must be 'full' or 'outline'")
        if mode == "outline":
            path, text, truncated = read_text_file(root, relative, max_chars=20_000)
            return _outline(path.relative_to(root).as_posix(), text, truncated=truncated)
        max_chars = optional_int(args, "max_chars", 8_000, upper=12_000)
        path, text, truncated = read_text_file(root, relative, max_chars=max_chars)
        suffix = "\n\n[truncated]" if truncated else ""
        shown = text.rstrip()
        return f"file: {path.relative_to(root).as_posix()}\n\n{shown}{suffix}"

    return Tool(
        name="read_file",
        description=(
            "Read a Markdown or text file inside the workspace. "
            "Use mode 'outline' for a summary (word count, headings, excerpt) "
            "and mode 'full' for the file text. "
            "path is relative to the workspace, for example notes/welcome.md."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to a .md, .markdown, or .txt file.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full", "outline"],
                    "description": (
                        "outline returns headings, word count, and a short excerpt. "
                        "full returns the file text. Use outline when asked to summarise."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters returned in full mode. Defaults to 8000.",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def _outline(relative: str, text: str, *, truncated: bool) -> str:
    headings: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            headings.append(f"{match.group(1)} {match.group(2)[:180]}")
    excerpt = text.strip()
    if len(excerpt) > 800:
        excerpt = excerpt[:800].rstrip() + "..."
    word_line = f"words: {len(text.split())}"
    if truncated:
        word_line += " (file truncated)"
    heading_block = "\n".join(headings) if headings else "(none)"
    return f"file: {relative}\n{word_line}\nheadings:\n{heading_block}\n\nexcerpt:\n{excerpt}"
