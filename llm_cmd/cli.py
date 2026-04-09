import argparse
import os
import sys

from .config import DEFAULT_MODEL
from .constants import CODE_SYSTEM_PROMPT
from .db import _UsageStats
from .models import _load_models
from .multimodal import _build_user_content


def _execute_prompt() -> str:
    shell = os.environ.get("SHELL", "/bin/bash")
    shell = os.path.basename(shell)
    return (
        f"You are a shell command generator for {shell}. "
        "Output ONLY a single executable shell command that accomplishes the user's request. "
        "No explanation. No markdown. No code fences. No newlines. "
        "Chain multiple steps with && or semicolons if needed."
    )


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
