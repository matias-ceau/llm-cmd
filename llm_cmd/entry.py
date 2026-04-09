import argparse
import os
import sys

from . import constants
from .cli import _execute_prompt, _print_stats, build_parser, get_content
from .config import DEFAULT_MODEL, _load_config, _resolve_default_model, _save_config
from .constants import CODE_SYSTEM_PROMPT
from .db import _record_message, _record_usage, _resolve_session
from .execute import confirm_and_run
from .http_client import call_llm_capture, call_llm_streaming
from .models import (
    _check_modality_support,
    _fetch_models,
    _list_models_by_modality,
    _load_models,
    _maybe_update_models_bg,
)


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

    def _post(response_text: str, stats, mode: str) -> None:
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


def main_status() -> None:
    cfg = _load_config()
    model = cfg.get("default_model") or DEFAULT_MODEL
    model_source = "config" if cfg.get("default_model") else (
        "env" if os.environ.get("LLM_CMD_MODEL") else "default"
    )
    n_models = len(_load_models())
    print(f"model         : {model}  ({model_source})")
    print(f"api_url       : {constants._API_URL}")
    print(f"api_key       : {constants._API_KEY or '(not set)'}")
    print(f"models cached : {n_models} models  ({constants._MODELS_CACHE})")
    print(f"config file   : {constants._CONFIG_FILE}  ({'exists' if constants._CONFIG_FILE.exists() else 'not created'})")
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
