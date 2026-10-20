"""Command line: `agent chat` and `agent run`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from harbor import __version__
from harbor.agent import Agent, ToolTrace, TurnResult
from harbor.client import OllamaClient, OllamaError
from harbor.config import ConfigError, Settings, load_settings
from harbor.tools import build_registry

app = typer.Typer(
    name="agent",
    help="Local Ollama agent for notes, public pages, arithmetic, and a task list.",
    no_args_is_help=True,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"harbor {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Local Ollama agent for notes, public pages, arithmetic, and a task list."""


@app.command()
def run(
    task: str = typer.Argument(..., help="What you want the agent to do."),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model name."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Folder the file and task tools may use."
    ),
    host: str | None = typer.Option(None, "--host", help="Ollama base URL."),
    max_steps: int | None = typer.Option(
        None, "--max-steps", min=1, max=20, help="Model round-trips for this task."
    ),
) -> None:
    """Run one task and print the tools that were called."""
    if not task.strip():
        typer.echo("error: task is empty", err=True)
        raise typer.Exit(code=1)
    settings = _settings(model=model, workspace=workspace, host=host, max_steps=max_steps)
    result = _ask(_agent(settings), task)
    _echo_result(result)
    if not result.finished:
        raise typer.Exit(code=2)


@app.command()
def chat(
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model name."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Folder the file and task tools may use."
    ),
    host: str | None = typer.Option(None, "--host", help="Ollama base URL."),
    max_steps: int | None = typer.Option(
        None, "--max-steps", min=1, max=20, help="Model round-trips per message."
    ),
) -> None:
    """Talk to the agent. The conversation stays in this process."""
    settings = _settings(model=model, workspace=workspace, host=host, max_steps=max_steps)
    agent = _agent(settings)
    typer.echo(f"Harbor  model={settings.model}  workspace={settings.workspace}")
    typer.echo("Ask something, or type /exit.")
    while True:
        try:
            text = typer.prompt("you", prompt_suffix=" › ")
        except (EOFError, KeyboardInterrupt, typer.Abort):
            typer.echo("\nbye")
            raise typer.Exit() from None
        cleaned = text.strip()
        if not cleaned:
            continue
        if cleaned.lower() in {"/exit", "/quit", "exit", "quit"}:
            typer.echo("bye")
            raise typer.Exit()
        result = _ask(agent, cleaned)
        _echo_result(result)


@app.command()
def web(
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model name."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Folder the file and task tools may use."
    ),
    host: str | None = typer.Option(None, "--host", help="Ollama base URL."),
    max_steps: int | None = typer.Option(
        None, "--max-steps", min=1, max=20, help="Model round-trips per message."
    ),
    bind: str = typer.Option("127.0.0.1", "--bind", help="Local address for the web UI."),
    port: int = typer.Option(8765, "--port", min=1, max=65535, help="Port for the web UI."),
) -> None:
    """Open the local web UI. It uses the same agent loop as the CLI."""
    settings = _settings(model=model, workspace=workspace, host=host, max_steps=max_steps)
    if bind not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo(
            "warning: Harbor has no login. A non-local bind exposes the workspace tools.",
            err=True,
        )
    from harbor.web import create_app

    typer.echo(f"Harbor UI  http://{bind}:{port}")
    typer.echo(f"model={settings.model}  workspace={settings.workspace}")
    typer.echo("Press Ctrl+C to stop.")
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("error: uvicorn is not installed", err=True)
        raise typer.Exit(code=1) from exc
    uvicorn.run(create_app(settings), host=bind, port=port, log_level="info")


@app.command("tools")
def list_tools(
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace used to build the tool list."
    ),
) -> None:
    """List the tools the agent can call."""
    settings = _settings(workspace=workspace)
    for tool in build_registry(settings.workspace):
        typer.echo(f"{tool.name}\n    {tool.description}")


def _settings(
    *,
    model: str | None = None,
    workspace: Path | None = None,
    host: str | None = None,
    max_steps: int | None = None,
) -> Settings:
    try:
        return load_settings(model=model, workspace=workspace, host=host, max_steps=max_steps)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _agent(settings: Settings) -> Agent:
    client = OllamaClient(settings.ollama_host, settings.timeout_seconds)
    return Agent(
        client=client,
        model=settings.model,
        tools=build_registry(settings.workspace),
        max_steps=settings.max_steps,
    )


def _ask(agent: Agent, text: str) -> TurnResult:
    try:
        return agent.ask(text)
    except OllamaError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("\nstopped")
        raise typer.Exit(code=130) from None


def _echo_result(result: TurnResult) -> None:
    if result.traces:
        for trace in result.traces:
            typer.echo(format_trace(trace))
    else:
        typer.echo("tools  (none)")
    typer.echo("")
    typer.echo(result.answer)


def format_trace(trace: ToolTrace) -> str:
    try:
        rendered = json.dumps(trace.arguments, ensure_ascii=False, separators=(", ", ": "))
    except TypeError:
        rendered = str(trace.arguments)
    if len(rendered) > 160:
        rendered = rendered[:157] + "..."
    status = "ok" if trace.ok else "error"
    lines = [f"tool  {trace.name}  {status}  {rendered}"]
    preview = trace.result.strip().splitlines() or [""]
    for line in preview[:6]:
        lines.append(f"      {line if len(line) <= 120 else line[:117] + '...'}")
    extra = len(preview) - 6
    if extra > 0:
        lines.append(f"      ... ({extra} more lines)")
    return "\n".join(lines)
