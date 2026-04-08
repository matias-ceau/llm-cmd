import argparse
import http.client
import json
import os
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

CODE_SYSTEM_PROMPT = (
    "You are an expert programmer. Output ONLY raw code — no explanation, "
    "no markdown fences, no prose before or after."
)


def _execute_prompt() -> str:
    shell = Path(os.environ.get("SHELL", "/bin/bash")).name
    return (
        f"You are a shell command generator for {shell}. "
        "Output ONLY a single executable shell command that accomplishes the user's request. "
        "No explanation. No markdown. No code fences. No newlines. "
        "Chain multiple steps with && or semicolons if needed."
    )

# Provider config — override via env vars:
#   LLM_CMD_MODEL   — model name          (e.g. anthropic/claude-3-5-haiku)
#   LLM_CMD_API_KEY — API key             (falls back to OPENROUTER_API_KEY)
#   LLM_CMD_API_URL — full endpoint URL   (any OpenAI-compatible API)
_DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("LLM_CMD_MODEL", "openai/gpt-4o-mini")
_API_URL = os.environ.get("LLM_CMD_API_URL", _DEFAULT_API_URL)
_API_KEY = os.environ.get("LLM_CMD_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

# Model cache (XDG-aware, 12h TTL for background refresh)
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llm-cmd"
_MODELS_CACHE = _CACHE_DIR / "models.json"
_CACHE_TTL = 43200
_SSL_CTX = ssl.create_default_context()


# ── Model cache ──────────────────────────────────────────────────────────────

def _models_url() -> str:
    parsed = urlparse(_API_URL)
    base = parsed.path.removesuffix("/chat/completions")
    return f"{parsed.scheme}://{parsed.netloc}{base}/models"


def _load_models() -> list[str]:
    """Return cached model IDs (stale cache is fine — background refresh handles freshness)."""
    if not _MODELS_CACHE.exists():
        return []
    try:
        data = json.loads(_MODELS_CACHE.read_text())
        return sorted(m["id"] for m in data.get("data", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _maybe_update_models_bg() -> None:
    """Spawn a detached background process to refresh the model cache if older than 12h."""
    if os.environ.get("_LLM_CMD_BG_UPDATE"):
        return  # we ARE the background process, don't recurse
    try:
        if time.time() - _MODELS_CACHE.stat().st_mtime < _CACHE_TTL:
            return  # cache is fresh enough
    except FileNotFoundError:
        pass
    subprocess.Popen(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r});"
            f"from llm_cmd import _fetch_models; _fetch_models()",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "_LLM_CMD_BG_UPDATE": "1"},
    )


def _fetch_models() -> list[str]:
    """Fetch model list from provider, save to cache, return sorted IDs."""
    url = _models_url()
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc, context=_SSL_CTX)
    conn.request("GET", parsed.path, headers={"Authorization": f"Bearer {_API_KEY}"})
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"Failed to fetch models: HTTP {resp.status}", file=sys.stderr)
        return []
    data = json.loads(resp.read().decode())
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_CACHE.write_text(json.dumps(data))
    return sorted(m["id"] for m in data.get("data", []))


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cmd",
        description="Ask questions or run AI-generated shell commands — no quotes needed.",
        usage="%(prog)s [-e|-c] [-m MODEL] [-s SYSTEM] [words ...]",
    )
    parser.add_argument(
        "words",
        nargs="*",
        help="Prompt words (no quoting needed). Reads stdin if empty.",
    )
    parser.add_argument(
        "-e", "--execute",
        action="store_true",
        help="Execute mode: generate and run a shell command.",
    )
    parser.add_argument(
        "-c", "--code",
        action="store_true",
        help="Code mode: generate code and print to stdout.",
    )
    model_arg = parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Model to use (default: {DEFAULT_MODEL} or $LLM_CMD_MODEL).",
    )
    parser.add_argument(
        "-s", "--system",
        default=None,
        metavar="PROMPT",
        help="Override the system prompt.",
    )
    parser.add_argument(
        "--update-models",
        action="store_true",
        help="Fetch and cache the model list from the provider, then exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print cached model IDs (one per line) and exit.",
    )

    # Attach completer — substring match so "haiku" finds "anthropic/claude-3-5-haiku"
    try:
        import argcomplete

        def _model_completer(prefix, **_):
            return [m for m in _load_models() if not prefix or prefix in m]

        model_arg.completer = _model_completer  # type: ignore[attr-defined]
    except ImportError:
        pass

    return parser


def get_prompt(args: argparse.Namespace) -> str:
    if args.words:
        return " ".join(args.words)
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            return prompt
    print("Usage: llm-cmd [words ...]\n       echo 'question' | llm-cmd", file=sys.stderr)
    sys.exit(1)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _make_request(prompt: str, model: str, system: str | None, stream: bool) -> http.client.HTTPResponse:
    if not _API_KEY:
        print("Error: no API key. Set LLM_CMD_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    parsed = urlparse(_API_URL)
    conn = http.client.HTTPSConnection(parsed.netloc, context=_SSL_CTX, timeout=30)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({"model": model, "messages": messages, "stream": stream})
    try:
        conn.request("POST", parsed.path, body, {
            "Authorization": f"Bearer {_API_KEY}",
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


def call_llm_streaming(prompt: str, model: str, system: str | None) -> None:
    resp = _make_request(prompt, model, system, stream=True)
    while line := resp.readline():
        text = line.decode().strip()
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    print()


def call_llm_capture(prompt: str, model: str, system: str) -> str:
    resp = _make_request(prompt, model, system, stream=False)
    return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()


# ── Execute mode ──────────────────────────────────────────────────────────────

def _strip_fences(command: str) -> str:
    command = command.strip()
    for prefix in ("```bash", "```sh", "```"):
        if command.startswith(prefix):
            command = command[len(prefix):]
            break
    return command.removesuffix("```").strip()


def _edit_in_editor(command: str) -> str:
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
        f.write(command)
        tmpfile = f.name
    try:
        os.system(f'{editor} "{tmpfile}"')
        return Path(tmpfile).read_text().strip()
    finally:
        os.unlink(tmpfile)


def confirm_and_run(command: str) -> None:
    command = _strip_fences(command)
    while True:
        print(f"\n\033[1;32m$ {command}\033[0m", file=sys.stderr)
        try:
            choice = input("Run this command? [y/N/e] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(0)
        if choice in ("y", "yes"):
            sys.exit(subprocess.run(command, shell=True).returncode)
        elif choice == "e":
            command = _edit_in_editor(command)
        else:
            print("Aborted.", file=sys.stderr)
            sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()

    try:
        import argcomplete
        argcomplete.autocomplete(parser)  # exits immediately if completing
    except ImportError:
        pass

    _maybe_update_models_bg()  # fire-and-forget, no impact on startup time

    args = parser.parse_args()

    if args.update_models:
        models = _fetch_models()
        print(f"Cached {len(models)} models → {_MODELS_CACHE}" if models else "No models returned.")
        return

    if args.list_models:
        models = _load_models()
        if not models:
            print("No cache — run: llm-cmd --update-models", file=sys.stderr)
            sys.exit(1)
        print("\n".join(models))
        return

    prompt = get_prompt(args)

    if args.execute:
        confirm_and_run(call_llm_capture(prompt, args.model, args.system or _execute_prompt()))
    elif args.code:
        call_llm_streaming(prompt, args.model, args.system or CODE_SYSTEM_PROMPT)
    else:
        call_llm_streaming(prompt, args.model, args.system)


if __name__ == "__main__":
    main()
