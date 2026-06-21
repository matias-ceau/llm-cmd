import argparse
import os
import sys

from . import constants
from .cli import _execute_prompt, _print_stats, build_parser, get_content
from .config import _ensure_config, _load_config, _resolve_default_model, _save_config
from .constants import CODE_SYSTEM_PROMPT
from .context import _machine_context
from .db import _record_message, _record_usage, _resolve_session
from .execute import confirm_and_run
from .http_client import call_llm_capture, call_llm_streaming
from .models import (
    _check_modality_support,
    _fetch_models,
    _list_models_by_modality,
    _load_models,
    _maybe_update_models_bg,
    _resolve_model_name,
)


def main() -> None:
    parser = build_parser()

    try:
        import argcomplete
        argcomplete.autocomplete(parser)  # exits immediately if completing
    except ImportError:
        pass

    _maybe_update_models_bg()  # fire-and-forget, no impact on startup time
    _ensure_config()  # creates ~/.config/llm-cmd/config.json on first run

    args = parser.parse_args()

    if args.tldr:
        from .docs import _TLDR
        print(_TLDR)
        return

    if args.docs:
        from .docs import _DOCS
        print(_DOCS)
        return

    if args.update_models:
        models = _fetch_models()
        print(f"Cached {len(models)} models → {constants._MODELS_CACHE}" if models else "No models returned.")
        return

    if args.list_models:
        models = _load_models()
        if not models:
            print("No cache — run: llm-cmd --update-models", file=sys.stderr)
            sys.exit(1)
        print("\n".join(models))
        return

    resolved_model = _resolve_model_name(args.model)
    if resolved_model != args.model:
        print(f"\033[2mModel: {resolved_model}\033[0m", file=sys.stderr)
    args.model = resolved_model

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

    def _default_system(mode_specific: str | None) -> str | None:
        cfg = _load_config()
        parts = [p for p in (mode_specific, _machine_context(), cfg.get("system_prompt")) if p]
        return "\n\n".join(parts) if parts else None

    def _build_messages(system: str | None) -> list[dict]:
        msgs = list(ctx_messages)
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _post(response_text: str, stats, mode: str) -> None:
        if stats:
            _record_usage(stats, mode)
        if show_stats:
            _print_stats(stats)
        if session_id:
            _record_message(session_id, "user",      user_content,    None,        None,  mode)
            _record_message(session_id, "assistant", response_text,   args.model,  stats, mode)

    if args.execute:
        msgs = _build_messages(args.system or _default_system(_execute_prompt()))
        cmd, stats = call_llm_capture(msgs, args.model)
        _post(cmd, stats, "execute")
        confirm_and_run(cmd, prompt_text)
    elif args.code:
        msgs = _build_messages(args.system or _default_system(CODE_SYSTEM_PROMPT))
        text, stats = call_llm_streaming(
            msgs,
            args.model,
            collect_usage=True,
            render_markdown=False,
        )
        _post(text, stats, "code")
    else:
        msgs = _build_messages(args.system or _default_system(None))
        text, stats = call_llm_streaming(
            msgs,
            args.model,
            collect_usage=True,
            render_markdown=True,
        )
        _post(text, stats, "chat")


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
    set_p = sub.add_parser(
        "set", help="Set the default model (omit MODEL for an interactive picker)."
    )
    set_p.add_argument(
        "model", nargs="?", default=None,
        help="Model ID, or a substring matching exactly one cached model. "
             "Omit to pick interactively from the cache.",
    )
    sub.add_parser("edit", help="Open the config file in $EDITOR.")

    args = parser.parse_args()
    _ensure_config()

    color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    marker = "\033[1;32m*\033[0m" if color else "*"

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

        current = _resolve_default_model()
        for m in models:
            print(f"{marker} {m}" if m == current else f"  {m}")

    elif args.cmd == "get":
        cfg = _load_config()
        source = "config" if cfg.get("default_model") else (
            "env" if os.environ.get("LLM_CMD_MODEL") else "default"
        )
        model = _resolve_default_model()
        name = f"\033[1m{model}\033[0m" if color else model
        print(f"{name}  ({source})")

    elif args.cmd == "set":
        model = args.model
        if model is None:
            models = _load_models()
            if not models:
                print("No cache — run: llm-cmd --update-models", file=sys.stderr)
                sys.exit(1)
            current = _resolve_default_model()
            for i, m in enumerate(models, 1):
                prefix = marker if m == current else " "
                print(f"{prefix} {i:3d}  {m}")
            try:
                choice = input("Select model number or name: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                sys.exit(0)
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                model = models[int(choice) - 1]
            else:
                model = _resolve_model_name(choice)
        else:
            model = _resolve_model_name(model)
        cfg = _load_config()
        cfg["default_model"] = model
        _save_config(cfg)
        print(f"Default model set to: {model}")

    elif args.cmd == "edit":
        editor = os.environ.get("EDITOR", "vi")
        os.system(f"{editor} {constants._CONFIG_FILE}")


def main_status() -> None:
    _ensure_config()
    cfg = _load_config()
    model = _resolve_default_model()
    model_source = "config" if cfg.get("default_model") else (
        "env" if os.environ.get("LLM_CMD_MODEL") else "default"
    )
    n_models = len(_load_models())
    print(f"model         : {model}  ({model_source})")
    print(f"api_url       : {constants._API_URL}")
    print(f"api_key       : {constants._API_KEY or '(not set)'}")
    print(f"models cached : {n_models} models  ({constants._MODELS_CACHE})")
    print(f"config file   : {constants._CONFIG_FILE}  ({'exists' if constants._CONFIG_FILE.exists() else 'not created'})")
    print(f"system_prompt : {cfg.get('system_prompt') or '(none)'}")
    print(f"machine ctx   : {_machine_context()}")
    print(f"history db    : {constants._HISTORY_DB}  ({'exists' if constants._HISTORY_DB.exists() else 'not created'})")


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

    from .db import _cost_summary
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
