"""Registry of tools the agent is allowed to call."""

from __future__ import annotations

from pathlib import Path

from harbor.sandbox import workspace_root
from harbor.tools.base import Tool
from harbor.tools.calc import make_calculate
from harbor.tools.fetch import make_fetch_url
from harbor.tools.files import make_read_file
from harbor.tools.tasks import make_manage_tasks

__all__ = ["Tool", "build_registry"]


def build_registry(workspace: Path) -> list[Tool]:
    root = workspace_root(workspace)
    return [
        make_read_file(root),
        make_fetch_url(),
        make_calculate(),
        make_manage_tasks(root),
    ]
