# Copilot instructions for `llm-cmd`

## Build, test, and lint commands

- **Install as local tool:** `uv tool install -e .`
- **Run CLI in dev mode (no install):** `uv run python -m llm_cmd <prompt>`
- **Run full tests:** `uv run pytest tests/ -v`
- **Run a single test:** `uv run pytest tests/test_llm_cmd.py::TestMakeRequest::test_http_error_exits -v`
- **Lint/build:** no dedicated lint or build command is configured in this repository.

## High-level architecture

`llm-cmd` is a minimal Python CLI with one main runtime path in `llm_cmd/entry.py`:

1. Parse args in `cli.build_parser()`, resolve prompt input in `cli.get_content()` (words, stdin, and multimodal files via `multimodal.py`).
2. Resolve session context from SQLite (`db._resolve_session`) and validate modality support against cached model metadata (`models._check_modality_support`).
3. Call the provider through the low-level HTTP layer in `http_client.py`:
   - streaming path (`call_llm_streaming`) for chat/code
   - capture path (`call_llm_capture`) for execute mode
4. Persist usage + messages in SQLite (`db.py`), print usage stats (unless quiet/non-TTY), and in execute mode confirm/run via `execute.confirm_and_run`.

Supporting subsystems:

- **Model cache** (`models.py`): cached at `~/.cache/llm-cmd/models.json`, background refresh with detached subprocess every 12h (`_maybe_update_models_bg`).
- **Config** (`config.py`): default model precedence is `LLM_CMD_MODEL` env var > config file (`~/.config/llm-cmd/config.json`) > hardcoded fallback.
- **History/sessions** (`db.py`): SQLite at `~/.local/share/llm-cmd/history.db` with `history` and `messages` tables.
- **Public API surface** (`__init__.py`): re-exports internals used by tests/backward compatibility.

## Key conventions (repo-specific)

- **Patchable globals convention:** constants like `_API_KEY`, `_API_URL`, `_MODELS_CACHE`, `_CONFIG_FILE`, `_HISTORY_DB`, `_CACHE_TTL` live in `constants.py` and should be accessed as module-qualified `constants._X` inside implementation modules (not imported directly) so tests can patch `llm_cmd.constants._X`.
- **Documentation-update convention for behavior changes:** update all four locations together:
  1. `llm_cmd/docs.py` (`_TLDR`)
  2. `llm_cmd/docs.py` (`_DOCS`)
  3. `README.md`
  4. `CLAUDE.md` architecture section
- **Execute-mode output contract:** command-generation responses are expected to be raw executable shell commands (no markdown/prose/newlines). `execute._strip_fences()` is a fallback sanitizer, not the primary contract.
- **HTTP/client dependency convention:** keep provider calls in `http_client.py` using `http.client` (no requests/httpx dependency introduced by default).
