import os
import ssl
from pathlib import Path

CODE_SYSTEM_PROMPT = (
    "You are an expert programmer. Output ONLY raw code — no explanation, "
    "no markdown fences, no prose before or after."
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
