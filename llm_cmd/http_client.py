import http.client
import json
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


def call_llm_streaming(
    messages: list[dict],
    model: str,
    collect_usage: bool = False,
) -> tuple[str, _UsageStats | None]:
    resp = _make_request(messages, model, stream=True, include_usage=collect_usage)
    usage_data: dict | None = None
    parts: list[str] = []
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
                    print(delta, end="", flush=True)
                    parts.append(delta)
            if collect_usage and "usage" in chunk:
                usage_data = chunk["usage"]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
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
