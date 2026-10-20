"""Fetch a public URL and return visible text. Private addresses are refused."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from harbor.sandbox import ToolError
from harbor.tools.base import Tool
from harbor.tools.common import optional_int, require_str

USER_AGENT = "HarborAgent/0.1 (local CLI; educational portfolio project)"
MAX_BYTES = 500_000
MAX_REDIRECTS = 3
_REDIRECTS = {301, 302, 303, 307, 308}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}

Resolver = Callable[..., list[Any]]


def make_fetch_url() -> Tool:
    def handler(args: dict[str, Any]) -> str:
        url = require_str(args, "url")
        max_chars = optional_int(args, "max_chars", 8_000, upper=12_000)
        return fetch_public_text(url, max_chars=max_chars)

    return Tool(
        name="fetch_url",
        description=(
            "Fetch a public http or https URL and return the visible text. "
            "Refuses private, loopback, and link-local addresses. "
            "Use this for public pages. It does not run JavaScript or log in."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Public http or https URL."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of extracted text. Defaults to 8000.",
                },
            },
            "required": ["url"],
        },
        handler=handler,
    )


def fetch_public_text(
    url: str,
    *,
    max_chars: int = 8_000,
    client: httpx.Client | None = None,
    resolve: Resolver = socket.getaddrinfo,
) -> str:
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=20.0, follow_redirects=False, trust_env=False)
    try:
        return _fetch(client, url, max_chars=max_chars, resolve=resolve)
    except httpx.TimeoutException as exc:
        raise ToolError("the request timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError("the request failed before a response arrived") from exc
    finally:
        if owns_client:
            client.close()


def validate_public_url(url: str, *, resolve: Resolver = socket.getaddrinfo) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ToolError("url is required")
    candidate = url.strip()
    if len(candidate) > 2_000:
        raise ToolError("url is too long")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("only http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise ToolError("URLs with credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise ToolError("url is missing a host")
    lowered = host.lower().rstrip(".")
    if lowered in _BLOCKED_HOSTS or lowered.endswith(_BLOCKED_SUFFIXES):
        raise ToolError(f"host is not public: {host}")
    if host.isdigit():
        raise ToolError("numeric hosts are not allowed")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _reject_ip(literal)
        return candidate
    try:
        records = resolve(host, None)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve host: {host}") from exc
    if not records:
        raise ToolError(f"could not resolve host: {host}")
    for record in records:
        address = str(record[4][0]).split("%", 1)[0]
        _reject_ip(ipaddress.ip_address(address))
    return candidate


def extract_text(content_type: str, body: bytes, *, max_chars: int) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime and not _is_text_mime(mime):
        raise ToolError(f"unsupported content type: {mime}")
    charset = _charset(content_type) or "utf-8"
    try:
        text = body.decode(charset, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    if "\x00" in text[:2_000]:
        raise ToolError("response looks like a binary file")
    if mime in {"text/html", "application/xhtml+xml"} or (not mime and _looks_like_html(text)):
        text = html_to_text(text)
    text = _collapse(text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[truncated]"
    return text.strip() or "(no text extracted)"


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    title = " ".join(part for part in parser.title_parts if part).strip()
    body = _collapse("".join(parser.parts))
    if title:
        return f"title: {title}\n\n{body}".strip()
    return body


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data.strip())
        if self._skip:
            return
        chunk = " ".join(data.split())
        if chunk:
            self.parts.append(chunk)
            self.parts.append(" ")


def _fetch(
    client: httpx.Client,
    url: str,
    *,
    max_chars: int,
    resolve: Resolver,
) -> str:
    current = url.strip()
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"}
    for hop in range(MAX_REDIRECTS + 1):
        current = validate_public_url(current, resolve=resolve)
        with client.stream("GET", current, headers=headers) as response:
            if response.status_code in _REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    raise ToolError("redirect was missing a location")
                if hop == MAX_REDIRECTS:
                    raise ToolError("too many redirects")
                current = urljoin(current, location)
                continue
            if not 200 <= response.status_code < 300:
                raise ToolError(f"request failed with status {response.status_code}")
            body = _read_capped(response)
            content_type = response.headers.get("content-type", "")
        return extract_text(content_type, body, max_chars=max_chars)
    raise ToolError("too many redirects")


def _read_capped(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise ToolError("response is too large")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_BYTES:
            raise ToolError("response is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not ip.is_global:
        raise ToolError(f"blocked non-public address: {ip}")


def _is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }


def _looks_like_html(text: str) -> bool:
    sample = text.lstrip()[:200].lower()
    return sample.startswith("<!doctype html") or sample.startswith("<html")


def _charset(content_type: str) -> str | None:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip().strip('"').strip("'")
    return None


def _collapse(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        compact = " ".join(line.split())
        if compact:
            lines.append(compact)
        elif lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()
