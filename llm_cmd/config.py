import json
import os

from . import constants

_HARDCODED_DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load_config() -> dict:
    if not constants._CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(constants._CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(data: dict) -> None:
    constants._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    constants._CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _ensure_config() -> dict:
    """Create the config file with current defaults on first run, so it
    always exists on disk and can be hand-edited in place."""
    cfg = _load_config()
    if not constants._CONFIG_FILE.exists():
        cfg.setdefault("default_model", os.environ.get("LLM_CMD_MODEL") or _HARDCODED_DEFAULT_MODEL)
        _save_config(cfg)
    return cfg


def _resolve_default_model() -> str:
    return (
        os.environ.get("LLM_CMD_MODEL")
        or _load_config().get("default_model")
        or _HARDCODED_DEFAULT_MODEL
    )


DEFAULT_MODEL = _resolve_default_model()
