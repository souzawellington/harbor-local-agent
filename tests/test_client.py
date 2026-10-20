import json

import httpx
import pytest

from harbor.client import OllamaClient, OllamaError


def test_chat_parses_tool_call_and_sends_documented_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "add the numbers",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": {"expression": "2+2"},
                            },
                        }
                    ],
                }
            },
        )

    client = OllamaClient(
        "http://127.0.0.1:11434/",
        timeout=5,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    message = client.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "calculate"}}],
    )

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["stream"] is False
    assert body["model"] == "llama3.2:3b"
    assert body["options"] == {"temperature": 0}
    assert message.tool_calls[0].name == "calculate"
    assert message.tool_calls[0].arguments == {"expression": "2+2"}
    assert message.thinking == "add the numbers"
    assert message.raw_tool_calls is not None


def test_arguments_may_be_a_json_string() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}}
                    ],
                }
            },
        )

    client = OllamaClient(
        "http://127.0.0.1:11434",
        timeout=5,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    message = client.chat(model="llama3.2:3b", messages=[], tools=[])
    assert message.tool_calls[0].arguments == {"expression": "2+2"}


def test_connection_error_names_the_next_step() -> None:
    class Refuse(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

    client = OllamaClient(
        "http://127.0.0.1:11434",
        timeout=5,
        http=httpx.Client(transport=Refuse()),
    )
    with pytest.raises(OllamaError, match="ollama pull llama3.2:3b"):
        client.chat(model="llama3.2:3b", messages=[], tools=[])


def test_missing_model_suggests_pull() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    client = OllamaClient(
        "http://127.0.0.1:11434",
        timeout=5,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OllamaError, match="ollama pull"):
        client.chat(model="missing:3b", messages=[], tools=[])


def test_non_json_response_is_an_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = OllamaClient(
        "http://127.0.0.1:11434",
        timeout=5,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OllamaError, match="not JSON"):
        client.chat(model="llama3.2:3b", messages=[], tools=[])
