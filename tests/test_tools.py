import socket
from pathlib import Path

import httpx
import pytest

from harbor.sandbox import ToolError, read_text_file
from harbor.tools import build_registry
from harbor.tools.base import Tool
from harbor.tools.calc import evaluate
from harbor.tools.fetch import extract_text, fetch_public_text, html_to_text
from harbor.tools.files import make_read_file
from harbor.tools.tasks import make_manage_tasks

REPO_WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"


def _public_resolve(host: str, port: int | None, *_args: object, **_kwargs: object) -> list[object]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _private_resolve(
    host: str, port: int | None, *_args: object, **_kwargs: object
) -> list[object]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]


def test_registry_schemas_are_unique_and_shaped(tmp_path: Path) -> None:
    tools = build_registry(tmp_path)
    names = [tool.name for tool in tools]
    assert names == ["read_file", "fetch_url", "calculate", "manage_tasks"]
    for tool in tools:
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"
        assert isinstance(schema["function"]["parameters"]["required"], list)


def test_handler_exception_becomes_a_tool_error() -> None:
    def handler(_args: dict[str, object]) -> str:
        raise RuntimeError("boom")

    tool = Tool("broken", "breaks", {"type": "object", "properties": {}}, handler)
    ok, result = tool.invoke({})
    assert ok is False
    assert result == "error: RuntimeError: boom"


def test_read_file_returns_text_and_outline(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "brief.md").write_text("# Brief\n\nA short note about APIs.\n", encoding="utf-8")
    tool = make_read_file(tmp_path)
    ok, full = tool.invoke({"path": "notes/brief.md", "mode": "full"})
    assert ok is True
    assert "A short note about APIs." in full
    ok, outline = tool.invoke({"path": "notes/brief.md", "mode": "summarise"})
    assert ok is True
    assert "file: notes/brief.md" in outline
    assert "# Brief" in outline
    assert "words:" in outline


def test_read_file_refuses_to_leave_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    tool = make_read_file(workspace)
    expected = {
        "../secret.md": "error: path is outside the workspace",
        "notes/../../secret.md": "error: path is outside the workspace",
        "/etc/passwd": "error: path must stay inside the workspace",
        "~/secret.md": "error: path must stay inside the workspace",
    }
    for path, message in expected.items():
        assert tool.invoke({"path": path}) == (False, message)


def test_symlink_outside_the_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "leak.md").symlink_to(outside)
    ok, result = make_read_file(workspace).invoke({"path": "leak.md"})
    assert ok is False
    assert "outside the workspace" in result
    assert "secret" not in result


def test_read_file_rejects_the_wrong_kind_of_file(tmp_path: Path) -> None:
    (tmp_path / "notes.py").write_text("print('no')\n", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    tool = make_read_file(tmp_path)
    ok, result = tool.invoke({"path": "notes.py"})
    assert ok is False
    assert ".md" in result
    ok, result = tool.invoke({"path": "empty"})
    assert ok is False
    assert "not a file" in result
    ok, result = tool.invoke({"path": "missing.md"})
    assert ok is False
    assert "file not found" in result


def test_read_file_rejects_oversized_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("harbor.sandbox.MAX_FILE_BYTES", 20)
    (tmp_path / "big.md").write_text("word " * 20, encoding="utf-8")
    with pytest.raises(ToolError, match="too large"):
        read_text_file(tmp_path, "big.md", max_chars=100)


def test_sample_workspace_note_and_task_are_readable() -> None:
    tools = {tool.name: tool for tool in build_registry(REPO_WORKSPACE)}
    ok, outline = tools["read_file"].invoke({"path": "notes/welcome.md", "mode": "outline"})
    assert ok is True
    assert "# Workspace notes" in outline
    assert "AI automation" in outline
    ok, tasks = tools["manage_tasks"].invoke({"action": "list"})
    assert ok is True
    assert "Read notes/welcome.md with the agent" in tasks


def test_calculate_accepts_arithmetic_and_refuses_calls(tmp_path: Path) -> None:
    assert evaluate("(12 + 8) / 4") == "5"
    assert evaluate("2+2*3") == "8"
    assert evaluate("-3 + 10") == "7"
    assert evaluate("7 % 4") == "3"
    assert evaluate("10 // 3") == "3"
    assert evaluate("0.1+0.2") == "0.3"
    for expression in (
        "__import__('os').system('echo hi')",
        "open('x')",
        "(2).bit_length()",
        "nine + 1",
        "True",
        "",
        "9**9**9",
    ):
        with pytest.raises(ToolError):
            evaluate(expression)
    with pytest.raises(ToolError, match="division by zero"):
        evaluate("1/0")
    target = tmp_path / "pwned.txt"
    with pytest.raises(ToolError):
        evaluate(f"open({str(target)!r}, 'w').write('x')")
    assert not target.exists()


def test_tasks_round_trip(tmp_path: Path) -> None:
    tool = make_manage_tasks(tmp_path)
    ok, empty = tool.invoke({"action": "list"})
    assert ok is True
    assert empty == "No tasks."
    ok, added = tool.invoke({"action": "add", "title": "  Review   the README  "})
    assert ok is True
    assert added == "Added task 1: Review the README"
    ok, listed = tool.invoke({"action": "list"})
    assert listed == "1. [ ] Review the README"
    ok, again = tool.invoke({"action": "complete", "task_id": "1"})
    assert ok is True
    assert again.startswith("Completed task 1")
    ok, already = tool.invoke({"action": "complete", "task_id": 1})
    assert already == "Task 1 is already complete."
    ok, deleted = tool.invoke({"action": "delete", "task_id": 1})
    assert deleted.startswith("Deleted task 1")
    assert tool.invoke({"action": "list"})[1] == "No tasks."
    assert not (tmp_path / "tasks.json.tmp").exists()
    ok, missing = tool.invoke({"action": "delete", "task_id": 9})
    assert ok is False
    assert "no task" in missing
    ok, bad = tool.invoke({"action": "drop"})
    assert ok is False


def test_tasks_symlink_cannot_point_outside(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(
        '{"tasks": [{"id": 1, "title": "secret", "done": false}]}\n',
        encoding="utf-8",
    )
    (workspace / "tasks.json").symlink_to(outside)
    ok, result = make_manage_tasks(workspace).invoke({"action": "list"})
    assert ok is False
    assert "outside the workspace" in result
    assert "secret" not in result
    assert outside.read_text(encoding="utf-8").count("secret") == 1


def test_tasks_file_must_be_valid(tmp_path: Path) -> None:
    (tmp_path / "tasks.json").write_text("{", encoding="utf-8")
    ok, result = make_manage_tasks(tmp_path).invoke({"action": "list"})
    assert ok is False
    assert "not valid JSON" in result


def test_html_extraction_drops_script_and_style() -> None:
    html = (
        "<html><head><title>Docs</title><style>body{}</style></head>"
        "<body><script>alert(1)</script><h1>Pricing</h1><p>Public page.</p></body></html>"
    )
    text = html_to_text(html)
    assert "title: Docs" in text
    assert "Pricing" in text
    assert "Public page." in text
    assert "alert" not in text
    assert "body{}" not in text


def test_extract_text_truncates_and_rejects_binary() -> None:
    text = extract_text("text/plain; charset=utf-8", b"abcdef", max_chars=3)
    assert text.startswith("abc")
    assert "[truncated]" in text
    with pytest.raises(ToolError, match="unsupported content type"):
        extract_text("image/png", b"\x89PNG", max_chars=100)


def test_fetch_blocks_non_public_targets() -> None:
    blocked = [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.1.2.3/secret",
        "http://localhost/admin",
        "http://metadata.google.internal/",
        "http://printer.local/",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
        "http://2130706433/",
    ]
    for url in blocked:
        with pytest.raises(ToolError):
            fetch_public_text(url, resolve=_public_resolve)
    with pytest.raises(ToolError, match="non-public"):
        fetch_public_text("http://example.com/", resolve=_private_resolve)


def test_fetch_extracts_html_and_refuses_a_redirect_home() -> None:
    html = (
        b"<html><head><title>Docs</title></head>"
        b"<body><p>Hello</p><script>nope</script></body></html>"
    )

    def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "example.com"
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html)

    text = fetch_public_text(
        "http://example.com/docs",
        client=httpx.Client(transport=httpx.MockTransport(ok_handler), follow_redirects=False),
        resolve=_public_resolve,
    )
    assert "title: Docs" in text
    assert "Hello" in text
    assert "nope" not in text

    seen: list[str] = []

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    with pytest.raises(ToolError, match="non-public"):
        fetch_public_text(
            "https://example.com/start",
            client=httpx.Client(
                transport=httpx.MockTransport(redirect_handler),
                follow_redirects=False,
            ),
            resolve=_public_resolve,
        )
    assert seen == ["example.com"]


def test_fetch_caps_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harbor.tools.fetch.MAX_BYTES", 8)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"0123456789abcdef",
        )

    with pytest.raises(ToolError, match="too large"):
        fetch_public_text(
            "http://example.com/big",
            client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False),
            resolve=_public_resolve,
        )
