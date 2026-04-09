import json
import os

from . import constants


def _load_config() -> dict:
    if not constants._CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(constants._CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(data: dict) -> None:
    constants._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    constants._CONFIG_FILE.write_text(json.dumps(data, indent=2))


def _resolve_default_model() -> str:
    return (
        os.environ.get("LLM_CMD_MODEL")
        or _load_config().get("default_model")
        or "openai/gpt-4o-mini"
    )


DEFAULT_MODEL = _resolve_default_model()
