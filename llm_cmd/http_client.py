import http.client
import json
import os
import re
import sys
from urllib.parse import urlparse

from . import constants
from .db import _UsageStats


def _make_request(
    messages: list[dict],
    model: str,
    stream: bool,
    include_usage: bool = False,
) -> http.client.HTTPResponse:
    if not constants._API_KEY:
        print("Error: no API key. Set LLM_CMD_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    parsed = urlparse(constants._API_URL)
    conn = http.client.HTTPSConnection(parsed.netloc, context=constants._SSL_CTX, timeout=30)

    body_dict: dict = {"model": model, "messages": messages, "stream": stream}
    if stream and include_usage:
        body_dict["stream_options"] = {"include_usage": True}
    body = json.dumps(body_dict)

    try:
        conn.request("POST", parsed.path, body, {
            "Authorization": f"Bearer {constants._API_KEY}",
            "Content-Type": "application/json",
        })
        resp = conn.getresponse()
    except OSError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)
    if resp.status != 200:
        print(f"API error {resp.status}: {resp.read().decode()}", file=sys.stderr)
        sys.exit(1)
    return resp


class _MarkdownAnsiRenderer:
    _RESET = "\033[0m"
    _HEADING = "\033[1;36m"
    _CODE = "\033[38;5;150m"
    _INLINE_CODE = "\033[38;5;214m"
    _BOLD = "\033[1m"
    _LIST_MARKER = "\033[38;5;215m"
    _BLOCKQUOTE = "\033[2;3m"

    _LIST_RE = re.compile(r"\d+[.)](?= )")

    def __init__(self) -> None:
        self._carry = ""
        self._line_start = True
        self._line_leading_spaces = 0
        self._in_heading = False
        self._in_fenced_code = False
        self._fence_ticks = 0
        self._fence_colored = False
        self._in_inline_code = False
        self._in_bold = False
        self._in_blockquote = False

    def _render_text(self, text: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\n":
                if self._in_heading:
                    out.append(self._RESET)
                    self._in_heading = False
                if self._in_blockquote:
                    out.append(self._RESET)
                    self._in_blockquote = False
                out.append("\n")
                self._line_start = True
                self._line_leading_spaces = 0
                i += 1
                continue

            if self._line_start and ch == " " and self._line_leading_spaces < 3:
                out.append(ch)
                self._line_leading_spaces += 1
                i += 1
                continue

            if ch == "`" and (self._line_start or self._line_leading_spaces <= 3):
                j = i
                while j < len(text) and text[j] == "`":
                    j += 1
                tick_count = j - i
                if tick_count >= 3:
                    if self._in_heading:
                        out.append(self._RESET)
                        self._in_heading = False
                    if self._in_inline_code:
                        out.append(self._RESET)
                        self._in_inline_code = False
                    if self._in_bold:
                        out.append("\033[22m")
                        self._in_bold = False
                    out.append("`" * tick_count)
                    if not self._in_fenced_code:
                        line_end = text.find("\n", j)
                        if line_end == -1:
                            line_end = len(text)
                        info = text[j:line_end].strip().lower()
                        self._fence_colored = info not in {"md", "markdown"}
                        self._in_fenced_code = True
                        self._fence_ticks = tick_count
                        if self._fence_colored:
                            out.append(self._CODE)
                    elif tick_count >= self._fence_ticks:
                        self._in_fenced_code = False
                        self._fence_ticks = 0
                        if self._fence_colored:
                            out.append(self._RESET)
                            self._fence_colored = False
                    self._line_start = False
                    self._line_leading_spaces = 4
                    i = j
                    continue

            if self._in_fenced_code:
                out.append(ch)
                self._line_start = False
                self._line_leading_spaces = 4
                i += 1
                continue

            if self._line_start and ch in "-*+" and text[i + 1:i + 2] == " ":
                out.append(self._LIST_MARKER)
                out.append(ch)
                out.append(self._RESET)
                self._line_start = False
                self._line_leading_spaces = 4
                i += 1
                continue

            if self._line_start and ch.isdigit():
                m = self._LIST_RE.match(text, i)
                if m:
                    out.append(self._LIST_MARKER)
                    out.append(m.group(0))
                    out.append(self._RESET)
                    self._line_start = False
                    self._line_leading_spaces = 4
                    i = m.end()
                    continue

            if self._line_start and ch == ">":
                out.append(self._BLOCKQUOTE)
                out.append(ch)
                self._in_blockquote = True
                self._line_start = False
                self._line_leading_spaces = 4
                i += 1
                continue

            if self._line_start and ch == "#":
                j = i
                while j < len(text) and text[j] == "#" and (j - i) < 6:
                    j += 1
                if j < len(text) and text[j] == " ":
                    out.append(self._HEADING)
                    out.append(text[i:j + 1])
                    self._in_heading = True
                    self._line_start = False
                    self._line_leading_spaces = 4
                    i = j + 1
                    continue

            if ch == "`":
                if self._in_inline_code:
                    out.append("`")
                    out.append(self._RESET)
                    if self._in_bold:
                        out.append(self._BOLD)
                    self._in_inline_code = False
                else:
                    out.append(self._INLINE_CODE)
                    out.append("`")
                    self._in_inline_code = True
                self._line_start = False
                self._line_leading_spaces = 4
                i += 1
                continue

            if text.startswith("**", i):
                if self._in_bold:
                    out.append("**\033[22m")
                    self._in_bold = False
                else:
                    out.append("**" + self._BOLD)
                    self._in_bold = True
                self._line_start = False
                self._line_leading_spaces = 4
                i += 2
                continue

            out.append(ch)
            self._line_start = False
            self._line_leading_spaces = 4
            i += 1
        return "".join(out)

    def render(self, chunk: str) -> str:
        if not chunk:
            return ""
        text = self._carry + chunk
        last_newline = text.rfind("\n")
        if last_newline == -1:
            self._carry = text
            return ""
        safe = text[:last_newline + 1]
        self._carry = text[last_newline + 1:]
        return self._render_text(safe)

    def finish(self) -> str:
        tail = self._render_text(self._carry)
        self._carry = ""
        if (
            self._in_heading or self._in_fenced_code or self._in_inline_code
            or self._in_bold or self._in_blockquote
        ):
            self._in_heading = False
            self._in_fenced_code = False
            self._fence_ticks = 0
            self._fence_colored = False
            self._in_inline_code = False
            self._in_bold = False
            self._in_blockquote = False
            return tail + self._RESET
        return tail


def _use_markdown_rendering() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def call_llm_streaming(
    messages: list[dict],
    model: str,
    collect_usage: bool = False,
    render_markdown: bool = True,
) -> tuple[str, _UsageStats | None]:
    resp = _make_request(messages, model, stream=True, include_usage=collect_usage)
    usage_data: dict | None = None
    parts: list[str] = []
    renderer = _MarkdownAnsiRenderer() if render_markdown and _use_markdown_rendering() else None
    while line := resp.readline():
        text = line.decode().strip()
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0]["delta"].get("content", "")
                if delta:
                    rendered = renderer.render(delta) if renderer else delta
                    print(rendered, end="", flush=True)
                    parts.append(delta)
            if collect_usage and "usage" in chunk:
                usage_data = chunk["usage"]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    if renderer:
        tail = renderer.finish()
        if tail:
            print(tail, end="", flush=True)
    print()
    stats = None
    if collect_usage and usage_data:
        stats = _UsageStats(
            model=model,
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cost_usd=usage_data.get("cost"),
        )
    return "".join(parts), stats


def call_llm_capture(
    messages: list[dict],
    model: str,
) -> tuple[str, _UsageStats | None]:
    resp = _make_request(messages, model, stream=False)
    data = json.loads(resp.read().decode())
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage")
    stats = _UsageStats(
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cost_usd=usage.get("cost"),
    ) if usage else None
    return text, stats
