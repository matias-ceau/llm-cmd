import argparse
import http.client
import json
import os
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
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
_API_URL = os.environ.get("LLM_CMD_API_URL", _DEFAULT_API_URL)
_API_KEY = os.environ.get("LLM_CMD_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

# XDG paths
_CACHE_DIR  = Path(os.environ.get("XDG_CACHE_HOME",  Path.home() / ".cache"))  / "llm-cmd"
_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "llm-cmd"
_DATA_DIR   = Path(os.environ.get("XDG_DATA_HOME",   Path.home() / ".local" / "share")) / "llm-cmd"

_MODELS_CACHE = _CACHE_DIR  / "models.json"
_CONFIG_FILE  = _CONFIG_DIR / "config.json"
_HISTORY_DB   = _DATA_DIR   / "history.db"

_CACHE_TTL = 43200
_SSL_CTX   = ssl.create_default_context()


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(_CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2))


def _resolve_default_model() -> str:
    return (
        os.environ.get("LLM_CMD_MODEL")
        or _load_config().get("default_model")
        or "openai/gpt-4o-mini"
    )


DEFAULT_MODEL = _resolve_default_model()


# ── Usage stats ───────────────────────────────────────────────────────────────

@dataclass
class _UsageStats:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None


# ── History (SQLite) ──────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            ts               REAL    NOT NULL,
            model            TEXT    NOT NULL,
            prompt_tokens    INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd         REAL,
            mode             TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON history(ts)")
    conn.commit()
    return conn


def _record_usage(stats: _UsageStats, mode: str) -> None:
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO history(ts,model,prompt_tokens,completion_tokens,cost_usd,mode)"
                " VALUES(?,?,?,?,?,?)",
                (time.time(), stats.model, stats.prompt_tokens,
                 stats.completion_tokens, stats.cost_usd, mode),
            )
    except Exception:
        pass  # history is non-critical, never crash


def _cost_summary(days: int) -> dict:
    if not _HISTORY_DB.exists():
        return {}
    since = 0.0 if days == 0 else time.time() - days * 86400
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(cost_usd), SUM(prompt_tokens), SUM(completion_tokens)"
                " FROM history WHERE ts > ?",
                (since,),
            ).fetchone()
    except Exception:
        return {}
    return {
        "requests":       row[0] or 0,
        "cost_usd":       row[1] or 0.0,
        "prompt_tokens":  row[2] or 0,
        "completion_tokens": row[3] or 0,
    }


# ── Model cache ───────────────────────────────────────────────────────────────

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
    try:
        conn.request("GET", parsed.path, headers={"Authorization": f"Bearer {_API_KEY}"})
        resp = conn.getresponse()
    except OSError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return []
    if resp.status != 200:
        print(f"Failed to fetch models: HTTP {resp.status}", file=sys.stderr)
        return []
    data = json.loads(resp.read().decode())
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_CACHE.write_text(json.dumps(data))
    return sorted(m["id"] for m in data.get("data", []))


# ── CLI ───────────────────────────────────────────────────────────────────────

_TLDR = """\
llm-cmd — quick reference

  llm-cmd what is anagnorisis          free-text question (no quotes needed)
  llm-cmd -e update all cargo bins     generate + confirm + run a shell command
  llm-cmd -c write a merge sort        generate code to stdout
  llm-cmd -m anthropic/claude-3 ...    use a specific model
  llm-cmd -q ...                       suppress usage stats

  llm-cmd-model list                   list cached models (* = default)
  llm-cmd-model get                    print current default model
  llm-cmd-model set openai/gpt-4o      set default model (saved to config)
  llm-cmd-status                       show current configuration
  llm-cmd-cost [--period 1d|7d|30d]    show cost summary

  llm-cmd --update-models              refresh model cache from provider
  llm-cmd --tldr                       this cheatsheet
  llm-cmd --docs                       full documentation\
"""

_DOCS = """\
llm-cmd(1)                         User Commands                        llm-cmd(1)

NAME
    llm-cmd — minimal no-TUI CLI for LLMs

SYNOPSIS
    llm-cmd [-e|-c] [-m MODEL] [-s SYSTEM] [-q] [words ...]
    llm-cmd --update-models | --list-models | --tldr | --docs
    llm-cmd-model list | get | set MODEL
    llm-cmd-status
    llm-cmd-cost [--period PERIOD]

DESCRIPTION
    llm-cmd sends a prompt to an LLM and streams the response to stdout.
    No quoting is needed — words on the command line are joined into the prompt.
    Stdin is read as the prompt when no words are given.

    Default provider is OpenRouter (OPENROUTER_API_KEY). Any OpenAI-compatible
    API can be used via environment variables.

OPTIONS
    words               Prompt words, joined with spaces. Reads stdin if empty.

    -e, --execute       Execute mode: generate a shell command, confirm, then run.
                        Prompts [Y/n/e] — Y is default (Enter to confirm).
                        Press e to open the command in $EDITOR with context.

    -c, --code          Code mode: generate raw code to stdout (no prose).

    -m, --model MODEL   Model to use for this invocation.
                        Default: $LLM_CMD_MODEL, config file, or openai/gpt-4o-mini.

    -s, --system PROMPT Override the system prompt.

    -q, --quiet         Suppress post-response usage stats (model, tokens, cost).

    --update-models     Fetch and cache the model list from the provider, then exit.

    --list-models       Print cached model IDs (one per line) and exit.

    --tldr              Show quick-reference cheatsheet and exit.

    --docs              Show this documentation and exit.

SUBCOMMANDS
    llm-cmd-model list      List cached models. Current default is marked with *.
    llm-cmd-model get       Print the current default model.
    llm-cmd-model set MODEL Set the default model (written to config file).

    llm-cmd-status          Print current configuration: API URL, key, model,
                            paths to config, cache, and history database.

    llm-cmd-cost            Show usage cost summary.
      --period PERIOD       Period: 1d, 7d (default), 30d, or all.

ENVIRONMENT
    LLM_CMD_MODEL       Default model name.
    LLM_CMD_API_KEY     API key (takes priority over OPENROUTER_API_KEY).
    LLM_CMD_API_URL     Full endpoint URL (default: OpenRouter).
    OPENROUTER_API_KEY  OpenRouter API key (fallback).
    XDG_CACHE_HOME      Cache directory (default: ~/.cache).
    XDG_CONFIG_HOME     Config directory (default: ~/.config).
    XDG_DATA_HOME       Data directory (default: ~/.local/share).
    EDITOR              Editor for -e edit mode (default: vi).
    SHELL               Shell name used in execute-mode system prompt.

FILES
    ~/.config/llm-cmd/config.json       Persistent config (default model).
    ~/.cache/llm-cmd/models.json        Cached model list (12h TTL).
    ~/.local/share/llm-cmd/history.db   Usage history (SQLite).

USAGE STATS
    After each response, llm-cmd prints to stderr:
        [model | N tok | $0.0012]
    Suppressed with -q/--quiet or when stdout is not a TTY.

EXAMPLES
    llm-cmd what does SIGKILL do
    llm-cmd -e find all png files larger than 1mb and compress them
    llm-cmd -c -m anthropic/claude-3-5-haiku write a binary search in Go
    echo "explain this" | llm-cmd
    llm-cmd-model set openai/gpt-4o
    llm-cmd-cost --period 30d\
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cmd",
        description="Ask questions or run AI-generated shell commands — no quotes needed.",
        usage="%(prog)s [-e|-c] [-m MODEL] [-s SYSTEM] [-q] [words ...]",
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
        "-q", "--quiet",
        action="store_true",
        help="Suppress post-response usage stats.",
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
    parser.add_argument(
        "--tldr",
        action="store_true",
        help="Show quick-reference cheatsheet and exit.",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Show full documentation and exit.",
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

def _make_request(
    prompt: str, model: str, system: str | None, stream: bool,
    include_usage: bool = False,
) -> http.client.HTTPResponse:
    if not _API_KEY:
        print("Error: no API key. Set LLM_CMD_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    parsed = urlparse(_API_URL)
    conn = http.client.HTTPSConnection(parsed.netloc, context=_SSL_CTX, timeout=30)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body_dict: dict = {"model": model, "messages": messages, "stream": stream}
    if stream and include_usage:
        body_dict["stream_options"] = {"include_usage": True}
    body = json.dumps(body_dict)

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


def call_llm_streaming(
    prompt: str, model: str, system: str | None,
    collect_usage: bool = False,
) -> _UsageStats | None:
    resp = _make_request(prompt, model, system, stream=True, include_usage=collect_usage)
    usage_data: dict | None = None
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
            if collect_usage and "usage" in chunk:
                usage_data = chunk["usage"]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    print()
    if collect_usage and usage_data:
        return _UsageStats(
            model=model,
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cost_usd=usage_data.get("cost"),
        )
    return None


def call_llm_capture(
    prompt: str, model: str, system: str | None,
) -> tuple[str, _UsageStats | None]:
    resp = _make_request(prompt, model, system, stream=False)
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


# ── Execute mode ──────────────────────────────────────────────────────────────

def _strip_fences(command: str) -> str:
    command = command.strip()
    for prefix in ("```bash", "```sh", "```"):
        if command.startswith(prefix):
            command = command[len(prefix):]
            break
    return command.removesuffix("```").strip()


def _edit_in_editor(command: str, prompt: str) -> str:
    editor = os.environ.get("EDITOR", "vi")
    sep = "─" * 48
    header = (
        f"# Prompt: {prompt}\n"
        f"# {sep}\n"
        f"# Edit the command below. Lines starting with # are ignored.\n\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
        f.write(header + command + "\n")
        tmpfile = f.name
    try:
        os.system(f'{editor} "{tmpfile}"')
        lines = Path(tmpfile).read_text().splitlines()
        return "\n".join(l for l in lines if not l.startswith("#")).strip()
    finally:
        os.unlink(tmpfile)


def confirm_and_run(command: str, prompt: str) -> None:
    command = _strip_fences(command)
    while True:
        print(f"\n\033[1;32m$ {command}\033[0m", file=sys.stderr)
        try:
            choice = input("Run this command? [Y/n/e] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(0)
        if choice in ("y", "yes", ""):
            sys.exit(subprocess.run(command, shell=True).returncode)
        elif choice == "e":
            command = _edit_in_editor(command, prompt)
        else:
            print("Aborted.", file=sys.stderr)
            sys.exit(0)


# ── Stats display ─────────────────────────────────────────────────────────────

def _print_stats(stats: _UsageStats | None) -> None:
    if stats is None:
        return
    total_tok = stats.prompt_tokens + stats.completion_tokens
    parts = [stats.model]
    if total_tok:
        parts.append(f"{total_tok} tok")
    if stats.cost_usd is not None:
        parts.append(f"${stats.cost_usd:.4f}")
    print(f"\033[2m  [{' | '.join(parts)}]\033[0m", file=sys.stderr)


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

    if args.tldr:
        print(_TLDR)
        return

    if args.docs:
        print(_DOCS)
        return

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
    show_stats = not args.quiet and sys.stdout.isatty()

    if args.execute:
        cmd, stats = call_llm_capture(prompt, args.model, args.system or _execute_prompt())
        if stats:
            _record_usage(stats, "execute")
        if show_stats:
            _print_stats(stats)
        confirm_and_run(cmd, prompt)
    elif args.code:
        stats = call_llm_streaming(
            prompt, args.model, args.system or CODE_SYSTEM_PROMPT,
            collect_usage=True,
        )
        if stats:
            _record_usage(stats, "code")
        if show_stats:
            _print_stats(stats)
    else:
        stats = call_llm_streaming(
            prompt, args.model, args.system,
            collect_usage=True,
        )
        if stats:
            _record_usage(stats, "chat")
        if show_stats:
            _print_stats(stats)


# ── llm-cmd-model ─────────────────────────────────────────────────────────────

def main_model() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-cmd-model",
        description="Manage the default model for llm-cmd.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List cached models (* marks the current default).")
    sub.add_parser("get",  help="Print the current default model.")
    set_p = sub.add_parser("set", help="Set the default model.")
    set_p.add_argument("model", help="Model ID to set as default.")
    args = parser.parse_args()

    if args.cmd == "list":
        models = _load_models()
        if not models:
            print("No cache — run: llm-cmd --update-models", file=sys.stderr)
            sys.exit(1)
        current = _load_config().get("default_model") or DEFAULT_MODEL
        for m in models:
            print(f"* {m}" if m == current else f"  {m}")

    elif args.cmd == "get":
        cfg = _load_config()
        source = "config" if cfg.get("default_model") else "env/default"
        print(f"{cfg.get('default_model') or DEFAULT_MODEL}  ({source})")

    elif args.cmd == "set":
        cfg = _load_config()
        cfg["default_model"] = args.model
        _save_config(cfg)
        print(f"Default model set to: {args.model}")


# ── llm-cmd-status ────────────────────────────────────────────────────────────

def main_status() -> None:
    cfg = _load_config()
    model = cfg.get("default_model") or DEFAULT_MODEL
    model_source = "config" if cfg.get("default_model") else (
        "env" if os.environ.get("LLM_CMD_MODEL") else "default"
    )
    n_models = len(_load_models())
    print(f"model         : {model}  ({model_source})")
    print(f"api_url       : {_API_URL}")
    print(f"api_key       : {_API_KEY or '(not set)'}")
    print(f"models cached : {n_models} models  ({_MODELS_CACHE})")
    print(f"config file   : {_CONFIG_FILE}  ({'exists' if _CONFIG_FILE.exists() else 'not created'})")
    print(f"history db    : {_HISTORY_DB}  ({'exists' if _HISTORY_DB.exists() else 'not created'})")


# ── llm-cmd-cost ──────────────────────────────────────────────────────────────

def main_cost() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-cmd-cost",
        description="Show llm-cmd usage cost summary.",
    )
    parser.add_argument(
        "--period",
        choices=["1d", "7d", "30d", "all"],
        default="7d",
        metavar="PERIOD",
        help="Period: 1d, 7d (default), 30d, or all.",
    )
    args = parser.parse_args()

    days = {"1d": 1, "7d": 7, "30d": 30, "all": 0}[args.period]
    s = _cost_summary(days)

    if not s or s["requests"] == 0:
        print(f"No history for period: {args.period}")
        return

    total_tok = s["prompt_tokens"] + s["completion_tokens"]
    cost_str  = f"${s['cost_usd']:.4f}" if s["cost_usd"] else "n/a"

    print(f"Period      : {args.period}")
    print(f"Requests    : {s['requests']}")
    print(f"Tokens      : {total_tok:,}  (prompt: {s['prompt_tokens']:,} / completion: {s['completion_tokens']:,})")
    print(f"Cost        : {cost_str}")


if __name__ == "__main__":
    main()
