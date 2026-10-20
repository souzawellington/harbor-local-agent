from pathlib import Path

from harbor.agent import SYSTEM_PROMPT, Agent
from harbor.client import ModelMessage, ToolCall
from harbor.tools import build_registry
from tests.fakes import ScriptedClient


def _agent(client: ScriptedClient, tmp_path: Path, max_steps: int = 4) -> Agent:
    return Agent(
        client=client,
        model="llama3.2:3b",
        tools=build_registry(tmp_path),
        max_steps=max_steps,
    )


def test_final_answer_does_not_invent_a_tool_call(tmp_path: Path) -> None:
    client = ScriptedClient([ModelMessage("Hello.", [])])
    result = _agent(client, tmp_path).ask("hello")
    assert result.finished is True
    assert result.answer == "Hello."
    assert result.traces == []
    assert client.requests[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT


def test_tool_result_is_sent_back_to_the_model(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage(
                "I'll calculate.",
                [ToolCall("calculate", {"expression": "2+2"})],
            ),
            ModelMessage("The answer is 4.", []),
        ]
    )
    result = _agent(client, tmp_path).ask("what is 2+2?")
    assert result.finished is True
    assert result.answer == "The answer is 4."
    assert result.traces[0].ok is True
    assert result.traces[0].result == "4"
    follow_up = client.requests[1]["messages"]
    assert follow_up[-1] == {"role": "tool", "tool_name": "calculate", "content": "4"}
    assert follow_up[-2]["tool_calls"][0]["function"]["name"] == "calculate"
    names = [tool["function"]["name"] for tool in client.requests[0]["tools"]]
    assert names == ["read_file", "fetch_url", "calculate", "manage_tasks"]


def test_raw_tool_call_is_echoed_unchanged(tmp_path: Path) -> None:
    raw = [
        {
            "type": "function",
            "function": {"index": 0, "name": "calculate", "arguments": {"expression": "1+1"}},
        }
    ]
    client = ScriptedClient(
        [
            ModelMessage("", [ToolCall("calculate", {"expression": "1+1"})], raw_tool_calls=raw),
            ModelMessage("2.", []),
        ]
    )
    _agent(client, tmp_path).ask("1+1")
    assert client.requests[1]["messages"][-2]["tool_calls"] == raw


def test_unknown_tool_is_reported_back(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage("", [ToolCall("shell", {"command": "ls"})]),
            ModelMessage("I cannot run a shell.", []),
        ]
    )
    result = _agent(client, tmp_path).ask("run ls")
    assert result.traces[0].ok is False
    assert "unknown tool shell" in result.traces[0].result
    assert client.requests[1]["messages"][-1]["content"].startswith("error: unknown tool")


def test_refused_path_is_returned_as_a_tool_error(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage("", [ToolCall("read_file", {"path": "/etc/passwd"})]),
            ModelMessage("That path is outside the workspace.", []),
        ]
    )
    result = _agent(client, tmp_path).ask("read the password file")
    assert result.traces[0].ok is False
    assert "workspace" in result.traces[0].result
    assert (
        "outside" in client.requests[1]["messages"][-1]["content"]
        or "inside" in (client.requests[1]["messages"][-1]["content"])
    )


def test_several_tools_in_one_turn_run_in_order(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage(
                "",
                [
                    ToolCall("calculate", {"expression": "2+2"}),
                    ToolCall("calculate", {"expression": "3+3"}),
                ],
            ),
            ModelMessage("4 and 6.", []),
        ]
    )
    result = _agent(client, tmp_path).ask("two sums")
    assert [trace.result for trace in result.traces] == ["4", "6"]


def test_max_steps_stops_the_loop(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage("", [ToolCall("calculate", {"expression": "1+1"})]),
            ModelMessage("", [ToolCall("calculate", {"expression": "1+1"})]),
        ]
    )
    result = _agent(client, tmp_path, max_steps=2).ask("keep going")
    assert result.finished is False
    assert "Stopped after 2" in result.answer
    assert len(result.traces) == 2
    assert len(client.requests) == 2


def test_thinking_is_used_when_content_is_empty(tmp_path: Path) -> None:
    client = ScriptedClient([ModelMessage("", [], thinking="The note is short.")])
    result = _agent(client, tmp_path).ask("summarise")
    assert result.answer == "The note is short."


def test_chat_history_is_kept_across_asks(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            ModelMessage("One.", []),
            ModelMessage("Two.", []),
        ]
    )
    agent = _agent(client, tmp_path)
    agent.ask("first")
    agent.ask("second")
    user_turns = [message["content"] for message in agent.messages if message["role"] == "user"]
    assert user_turns == ["first", "second"]


def test_blank_input_does_not_call_the_model(tmp_path: Path) -> None:
    client = ScriptedClient([])
    result = _agent(client, tmp_path).ask("   ")
    assert result.answer == "Say what you want me to do."
    assert client.requests == []
