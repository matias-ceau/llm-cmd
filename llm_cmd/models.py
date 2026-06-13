import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from . import constants


def _models_url() -> str:
    parsed = urlparse(constants._API_URL)
    base = parsed.path.removesuffix("/chat/completions")
    return f"{parsed.scheme}://{parsed.netloc}{base}/models"


def _load_models() -> list[str]:
    """Return cached model IDs (stale cache is fine — background refresh handles freshness)."""
    if not constants._MODELS_CACHE.exists():
        return []
    try:
        data = json.loads(constants._MODELS_CACHE.read_text())
        return sorted(m["id"] for m in data.get("data", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _load_models_full() -> list[dict]:
    if not constants._MODELS_CACHE.exists():
        return []
    try:
        return json.loads(constants._MODELS_CACHE.read_text()).get("data", [])
    except (json.JSONDecodeError, OSError):
        return []


def _maybe_update_models_bg() -> None:
    """Spawn a detached background process to refresh the model cache if older than 12h."""
    if os.environ.get("_LLM_CMD_BG_UPDATE"):
        return  # we ARE the background process, don't recurse
    try:
        if time.time() - constants._MODELS_CACHE.stat().st_mtime < constants._CACHE_TTL:
            return  # cache is fresh enough
    except FileNotFoundError:
        pass
    pkg_parent = str(Path(__file__).parent.parent)
    subprocess.Popen(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {pkg_parent!r});"
            f"from llm_cmd.models import _fetch_models; _fetch_models()",
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
    conn = http.client.HTTPSConnection(parsed.netloc, context=constants._SSL_CTX)
    try:
        conn.request("GET", parsed.path, headers={"Authorization": f"Bearer {constants._API_KEY}"})
        resp = conn.getresponse()
    except OSError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return []
    if resp.status != 200:
        print(f"Failed to fetch models: HTTP {resp.status}", file=sys.stderr)
        return []
    data = json.loads(resp.read().decode())
    constants._CACHE_DIR.mkdir(parents=True, exist_ok=True)
    constants._MODELS_CACHE.write_text(json.dumps(data))
    return sorted(m["id"] for m in data.get("data", []))


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
