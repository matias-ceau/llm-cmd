import argparse
import base64
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

# Multimodal file extension → MIME type
_MEDIA_EXTENSIONS: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif",  ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".mp4": "video/mp4",  ".webm": "video/webm",
}
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})
_PDF_EXTS   = frozenset({".pdf"})
_AUDIO_EXTS = frozenset({".mp3", ".wav", ".ogg"})
_VIDEO_EXTS = frozenset({".mp4", ".webm"})
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB warning threshold


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
            ts                REAL    NOT NULL,
            model             TEXT    NOT NULL,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd          REAL,
            mode              TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON history(ts)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        TEXT    NOT NULL,
            ts                REAL    NOT NULL,
            role              TEXT    NOT NULL,
            content           TEXT    NOT NULL,
            model             TEXT,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd          REAL,
            mode              TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)")
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
        pass


def _record_message(
    session_id: str,
    role: str,
    content: str | list,
    model: str | None,
    stats: _UsageStats | None,
    mode: str | None,
) -> None:
    content_str = json.dumps(content) if isinstance(content, list) else content
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO messages"
                "(session_id,ts,role,content,model,prompt_tokens,completion_tokens,cost_usd,mode)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id, time.time(), role, content_str, model,
                    stats.prompt_tokens if stats else 0,
                    stats.completion_tokens if stats else 0,
                    stats.cost_usd if stats else None,
                    mode,
                ),
            )
    except Exception:
        pass


def _get_session_messages(session_id: str) -> list[dict]:
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY ts",
                (session_id,),
            ).fetchall()
    except Exception:
        return []
    result = []
    for role, content in rows:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                result.append({"role": role, "content": parsed})
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        result.append({"role": role, "content": content})
    return result


def _last_session_id() -> str | None:
    if not _HISTORY_DB.exists():
        return None
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT session_id FROM messages ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _resolve_session(
    session_arg: str | None,
    follow_up: bool,
) -> tuple[str | None, list[dict]]:
    if session_arg and follow_up:
        print("Error: --session and --follow-up are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if follow_up:
        sid = _last_session_id()
        if not sid:
            print("No previous session found.", file=sys.stderr)
            sys.exit(1)
        print(f"\033[2mSession: {sid}\033[0m", file=sys.stderr)
        return sid, _get_session_messages(sid)

    if session_arg is None:
        return None, []

    if session_arg == "auto":
        sid = "auto-" + time.strftime("%Y%m%dT%H%M%S")
        print(f"\033[2mSession: {sid}\033[0m", file=sys.stderr)
        return sid, []

    # Named session — load history if it exists
    msgs = _get_session_messages(session_arg)
    return session_arg, msgs


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
        "requests":          row[0] or 0,
        "cost_usd":          row[1] or 0.0,
        "prompt_tokens":     row[2] or 0,
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


def _load_models_full() -> list[dict]:
    if not _MODELS_CACHE.exists():
        return []
    try:
        return json.loads(_MODELS_CACHE.read_text()).get("data", [])
    except (json.JSONDecodeError, OSError):
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


# ── Multimodal ────────────────────────────────────────────────────────────────

def _is_image_url(token: str) -> bool:
    if not token.startswith(("http://", "https://")):
        return False
    return Path(urlparse(token).path).suffix.lower() in _IMAGE_EXTS


def _encode_file_content(path: Path) -> dict:
    if path.stat().st_size > _MAX_FILE_BYTES:
        mb = path.stat().st_size // 1024 // 1024
        print(f"Warning: {path.name} is {mb}MB — may exceed API limits.", file=sys.stderr)
    ext = path.suffix.lower()
    mime = _MEDIA_EXTENSIONS.get(ext, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode()
    if ext in _IMAGE_EXTS:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
    if ext in _PDF_EXTS:
        return {"type": "file", "file": {"filename": path.name, "data": data}}
    if ext in _AUDIO_EXTS:
        return {"type": "input_audio", "input_audio": {"data": data, "format": ext.lstrip(".")}}
    if ext in _VIDEO_EXTS:
        return {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}}
    return {"type": "file", "file": {"filename": path.name, "data": data}}


def _build_user_content(
    words: list[str],
    extra_files: list[str] | None = None,
) -> tuple[str | list[dict], set[str]]:
    """
    Scan words and extra_files for media files/URLs.
    Returns (content, detected_input_modalities).
    content is a plain str when text-only, list[dict] when multimodal.
    """
    text_tokens: list[str] = []
    media_parts: list[dict] = []
    detected: set[str] = set()

    def _add_file(path: Path) -> None:
        ext = path.suffix.lower()
        media_parts.append(_encode_file_content(path))
        if ext in _IMAGE_EXTS:   detected.add("image")
        elif ext in _PDF_EXTS:   detected.add("file")
        elif ext in _AUDIO_EXTS: detected.add("audio")
        elif ext in _VIDEO_EXTS: detected.add("video")

    for f in (extra_files or []):
        p = Path(f)
        if p.exists():
            _add_file(p)
        else:
            print(f"Warning: input file not found: {f}", file=sys.stderr)

    for word in words:
        p = Path(word)
        if p.suffix.lower() in _MEDIA_EXTENSIONS and p.exists():
            _add_file(p)
        elif _is_image_url(word):
            media_parts.append({"type": "image_url", "image_url": {"url": word}})
            detected.add("image")
        else:
            text_tokens.append(word)

    prompt_text = " ".join(text_tokens)
    if not media_parts:
        return prompt_text, detected

    content: list[dict] = []
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    content.extend(media_parts)
    return content, detected


def _load_model_arch(model_id: str) -> dict:
    for m in _load_models_full():
        if m.get("id") == model_id:
            return m.get("architecture", {})
    return {}


def _list_models_by_modality(
    in_mods: list[str] | None = None,
    out_mods: list[str] | None = None,
) -> list[str]:
    result = []
    for m in _load_models_full():
        arch = m.get("architecture", {})
        in_ok  = not in_mods  or all(x in arch.get("input_modalities",  []) for x in in_mods)
        out_ok = not out_mods or all(x in arch.get("output_modalities", []) for x in out_mods)
        if in_ok and out_ok:
            result.append(m["id"])
    return sorted(result)


def _check_modality_support(model: str, needed: set[str]) -> None:
    if not needed:
        return
    arch = _load_model_arch(model)
    if not arch:
        print(f"Warning: cannot verify modality support for {model} (cache miss).", file=sys.stderr)
        return
    supported = set(arch.get("input_modalities", []))
    missing = needed - supported
    if not missing:
        return
    print(
        f"Error: model {model!r} does not support input modality: {', '.join(sorted(missing))}",
        file=sys.stderr,
    )
    compatible = _list_models_by_modality(in_mods=list(needed))
    if compatible:
        print("Models that support this:", file=sys.stderr)
        for m in compatible[:10]:
            print(f"  {m}", file=sys.stderr)
        if len(compatible) > 10:
            print(f"  … and {len(compatible) - 10} more. Use: llm-cmd-model list --in {','.join(sorted(needed))}", file=sys.stderr)
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

_TLDR = """\
llm-cmd — quick reference

  llm-cmd what is anagnorisis          free-text question (no quotes needed)
  llm-cmd describe this photo.jpg      multimodal: auto-detect files in words
  llm-cmd -i photo.jpg what is this    multimodal: explicit file input
  llm-cmd -e update all cargo bins     generate + confirm + run a shell command
  llm-cmd -c write a merge sort        generate code to stdout
  llm-cmd -m anthropic/claude-3 ...    use a specific model
  llm-cmd -q ...                       suppress usage stats
  llm-cmd -s myconv ask something      start or continue a named session
  llm-cmd -s auto ask something        auto-named session (name shown on stderr)
  llm-cmd -f what did I just ask       follow-up on last session

  llm-cmd-model list                   list cached models (* = default)
  llm-cmd-model list --in image        models accepting image input
  llm-cmd-model list --out audio       models that can generate audio
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
    llm-cmd [-e|-c] [-m MODEL] [-S SYSTEM] [-s SESSION|-f] [-i FILE] [-q] [words ...]
    llm-cmd --update-models | --list-models | --tldr | --docs
    llm-cmd-model list [--in MODALITIES] [--out MODALITIES]
    llm-cmd-model get | set MODEL
    llm-cmd-status
    llm-cmd-cost [--period PERIOD]

DESCRIPTION
    llm-cmd sends a prompt to an LLM and streams the response to stdout.
    No quoting is needed — words on the command line are joined into the prompt.
    Files (images, PDFs, audio, video) are auto-detected by extension in the
    word list, or supplied explicitly with -i.

OPTIONS
    words               Prompt words, joined with spaces. Files detected by
                        extension are passed as multimodal content. Reads stdin
                        if no words and no -i files are given.

    -e, --execute       Execute mode: generate a shell command, confirm, then run.
                        Prompts [Y/n/e] — Y is default (Enter to confirm).
                        Press e to open the command in $EDITOR with context.

    -c, --code          Code mode: generate raw code to stdout (no prose).

    -m, --model MODEL   Model to use for this invocation.
                        Default: $LLM_CMD_MODEL, config file, or openai/gpt-4o-mini.

    -S, --system PROMPT Override the system prompt.

    -s, --session NAME  Attach to a named session. Use 'auto' to generate a
                        timestamped name (printed to stderr for reuse).

    -f, --follow-up     Continue the last session in history. Mutually exclusive
                        with --session. The session name is printed to stderr.

    -i, --input FILE    Explicitly pass a file as multimodal input. Repeatable.
                        Files are also auto-detected from words by extension.

    -q, --quiet         Suppress post-response usage stats (model, tokens, cost).

    --update-models     Fetch and cache the model list from the provider, then exit.

    --list-models       Print cached model IDs (one per line) and exit.

    --tldr              Show quick-reference cheatsheet and exit.

    --docs              Show this documentation and exit.

SUBCOMMANDS
    llm-cmd-model list [--in MOD,MOD] [--out MOD,MOD]
                        List cached models. Current default is marked with *.
                        --in: filter by required input modalities (comma-separated).
                        --out: filter by supported output modalities.
                        Valid modalities: text, image, audio, video, file.

    llm-cmd-model get   Print the current default model.
    llm-cmd-model set MODEL  Set the default model (written to config file).

    llm-cmd-status      Print current configuration: API URL, key, model,
                        paths to config, cache, and history database.

    llm-cmd-cost        Show usage cost summary.
      --period PERIOD   Period: 1d, 7d (default), 30d, or all.

MULTIMODAL
    Supported input formats:
      Images : .jpg .jpeg .png .gif .webp  (also accepts https:// image URLs)
      PDFs   : .pdf
      Audio  : .mp3 .wav .ogg
      Video  : .mp4 .webm

    If the selected model does not support the required input modality, llm-cmd
    prints an error and lists compatible models from the cache.

SESSIONS
    Sessions group messages into multi-turn conversations. Each exchange
    (without -s/-f) is a standalone interaction stored in history.

    llm-cmd -s myproject explain the architecture
    llm-cmd -s myproject what about the tests ?
    llm-cmd -f any other suggestions ?   # continues last session

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
    ~/.local/share/llm-cmd/history.db   Usage history + sessions (SQLite).

USAGE STATS
    After each response, llm-cmd prints to stderr:
        [model | N tok | $0.0012]
    Suppressed with -q/--quiet or when stdout is not a TTY.\
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cmd",
        description="Ask questions or run AI-generated shell commands — no quotes needed.",
        usage="%(prog)s [-e|-c] [-m MODEL] [-S SYSTEM] [-s SESSION|-f] [-i FILE] [-q] [words ...]",
    )
    parser.add_argument(
        "words",
        nargs="*",
        help="Prompt words (no quoting needed). Files auto-detected by extension.",
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
        "-S", "--system",
        default=None,
        metavar="PROMPT",
        help="Override the system prompt.",
    )
    parser.add_argument(
        "-s", "--session",
        default=None,
        metavar="SESSION",
        help="Session name or 'auto'. Displayed on stderr for reuse.",
    )
    parser.add_argument(
        "-f", "--follow-up",
        action="store_true",
        help="Continue the last session in history.",
    )
    parser.add_argument(
        "-i", "--input",
        action="append",
        metavar="FILE",
        help="Explicit file input (image/pdf/audio/video). Repeatable.",
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


def get_content(
    args: argparse.Namespace,
) -> tuple[str | list[dict], set[str]]:
    """
    Returns (user_content, detected_input_modalities).
    Handles stdin fallback, word joining, and multimodal file detection.
    """
    if args.words or args.input:
        return _build_user_content(args.words, args.input or [])
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text, set()
    print("Usage: llm-cmd [words ...]\n       echo 'question' | llm-cmd", file=sys.stderr)
    sys.exit(1)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _make_request(
    messages: list[dict],
    model: str,
    stream: bool,
    include_usage: bool = False,
) -> http.client.HTTPResponse:
    if not _API_KEY:
        print("Error: no API key. Set LLM_CMD_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    parsed = urlparse(_API_URL)
    conn = http.client.HTTPSConnection(parsed.netloc, context=_SSL_CTX, timeout=30)

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

    session_id, ctx_messages = _resolve_session(args.session, args.follow_up)
    user_content, detected_mods = get_content(args)

    # Prompt text for display / execute mode system prompt
    if isinstance(user_content, str):
        prompt_text = user_content
    else:
        prompt_text = " ".join(
            p["text"] for p in user_content if isinstance(p, dict) and p.get("type") == "text"
        )

    _check_modality_support(args.model, detected_mods)

    show_stats = not args.quiet and sys.stdout.isatty()

    def _build_messages(system: str | None) -> list[dict]:
        msgs = list(ctx_messages)
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _post(response_text: str, stats: _UsageStats | None, mode: str) -> None:
        if stats:
            _record_usage(stats, mode)
        if show_stats:
            _print_stats(stats)
        if session_id:
            _record_message(session_id, "user",      user_content,    None,        None,  mode)
            _record_message(session_id, "assistant", response_text,   args.model,  stats, mode)

    if args.execute:
        msgs = _build_messages(args.system or _execute_prompt())
        cmd, stats = call_llm_capture(msgs, args.model)
        _post(cmd, stats, "execute")
        confirm_and_run(cmd, prompt_text)
    elif args.code:
        msgs = _build_messages(args.system or CODE_SYSTEM_PROMPT)
        text, stats = call_llm_streaming(msgs, args.model, collect_usage=True)
        _post(text, stats, "code")
    else:
        msgs = _build_messages(args.system)
        text, stats = call_llm_streaming(msgs, args.model, collect_usage=True)
        _post(text, stats, "chat")


# ── llm-cmd-model ─────────────────────────────────────────────────────────────

def main_model() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-cmd-model",
        description="Manage the default model for llm-cmd.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List cached models (* marks the current default).")
    list_p.add_argument("--in",  dest="in_filter",  default=None, metavar="MODALITIES",
                        help="Comma-separated required input modalities (e.g. image,text).")
    list_p.add_argument("--out", dest="out_filter", default=None, metavar="MODALITIES",
                        help="Comma-separated required output modalities (e.g. audio).")

    sub.add_parser("get",  help="Print the current default model.")
    set_p = sub.add_parser("set", help="Set the default model.")
    set_p.add_argument("model", help="Model ID to set as default.")
    args = parser.parse_args()

    if args.cmd == "list":
        in_mods  = [x.strip() for x in args.in_filter.split(",")]  if args.in_filter  else None
        out_mods = [x.strip() for x in args.out_filter.split(",")] if args.out_filter else None

        if in_mods or out_mods:
            models = _list_models_by_modality(in_mods, out_mods)
            if not models:
                print("No models match the given modality filters.", file=sys.stderr)
                sys.exit(1)
        else:
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
