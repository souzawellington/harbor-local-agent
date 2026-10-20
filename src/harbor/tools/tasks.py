"""Task list stored as JSON inside the workspace. No other files are written."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.sandbox import ToolError
from harbor.tools.base import Tool
from harbor.tools.common import require_int, require_str

TASKS_FILENAME = "tasks.json"
_MAX_TITLE = 200


def make_manage_tasks(root: Path) -> Tool:
    def handler(args: dict[str, Any]) -> str:
        path = _safe_tasks_path(root)
        action = require_str(args, "action").lower()
        tasks = _load(path)
        if action == "list":
            return _render(tasks)
        if action == "add":
            title = " ".join(require_str(args, "title").split())
            if len(title) > _MAX_TITLE:
                raise ToolError("title is too long")
            next_id = max((item["id"] for item in tasks), default=0) + 1
            tasks.append(
                {
                    "id": next_id,
                    "title": title,
                    "done": False,
                    "created": datetime.now(UTC).strftime("%Y-%m-%d"),
                }
            )
            _save(path, tasks)
            return f"Added task {next_id}: {title}"
        if action in {"complete", "delete"}:
            task_id = require_int(args, "task_id")
            index = next((i for i, item in enumerate(tasks) if item["id"] == task_id), None)
            if index is None:
                raise ToolError(f"no task with id {task_id}")
            task = tasks[index]
            if action == "complete":
                if task["done"]:
                    return f"Task {task_id} is already complete."
                task["done"] = True
                _save(path, tasks)
                return f"Completed task {task_id}: {task['title']}"
            deleted = tasks.pop(index)
            _save(path, tasks)
            return f"Deleted task {task_id}: {deleted['title']}"
        raise ToolError("action must be list, add, complete, or delete")

    return Tool(
        name="manage_tasks",
        description=(
            "List, add, complete, or delete items in the workspace task list. "
            "action is list, add, complete, or delete. "
            "add requires title. complete and delete require task_id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "complete", "delete"],
                },
                "title": {
                    "type": "string",
                    "description": "Task title. Required when action is add.",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Task id. Required when action is complete or delete.",
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )


def _safe_tasks_path(root: Path) -> Path:
    root = root.resolve()
    try:
        path = (root / TASKS_FILENAME).resolve()
    except OSError as exc:
        raise ToolError("tasks.json path is invalid") from exc
    if not path.is_relative_to(root):
        raise ToolError("tasks.json is outside the workspace")
    return path


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"tasks.json is not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ToolError("tasks.json must be an object with a tasks array")
    items: list[dict[str, Any]] = []
    for item in data["tasks"]:
        if (
            not isinstance(item, dict)
            or type(item.get("id")) is not int
            or not isinstance(item.get("title"), str)
        ):
            raise ToolError("each task needs an integer id and a title")
        items.append(
            {
                "id": item["id"],
                "title": item["title"],
                "done": bool(item.get("done", False)),
                "created": str(item.get("created") or ""),
            }
        )
    return items


def _save(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "No tasks."
    lines = []
    for task in tasks:
        mark = "x" if task["done"] else " "
        lines.append(f"{task['id']}. [{mark}] {task['title']}")
    return "\n".join(lines)
