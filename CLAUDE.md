# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-cmd` is a minimal no-TUI Python CLI for LLMs. Core use cases:
- `llm-cmd what is the meaning of anagnorisis` — free-text prompt, no quotes
- `llm-cmd -e update all my cargo binaries` — generate + confirm + run a shell command
- `llm-cmd-model set openai/gpt-4o` — set persistent default model
- `llm-cmd-status` — show current configuration
- `llm-cmd-cost --period 30d` — usage cost summary

Default provider: OpenRouter (`OPENROUTER_API_KEY`). Any OpenAI-compatible API works via `LLM_CMD_API_URL` / `LLM_CMD_API_KEY` / `LLM_CMD_MODEL`.

## Install (Arch Linux / uv)

```bash
uv tool install -e .

# Activate tab-completion (add to .bashrc / .zshrc)
eval "$(register-python-argcomplete llm-cmd)"

# Populate model cache (auto-refreshes every 12h in background)
llm-cmd --update-models
```

## Development commands

```bash
uv run llm_cmd.py <prompt>       # run without installing
llm-cmd --list-models            # inspect model cache
llm-cmd --update-models          # force model cache refresh
```

## Architecture

Single-file implementation: `llm_cmd.py`

- **HTTP layer** (`_make_request`): direct `http.client` calls, zero third-party deps except `argcomplete`
- **Streaming** (`call_llm_streaming`): SSE parsed line-by-line, tokens printed as received; returns `_UsageStats | None`
- **Execute mode** (`confirm_and_run`): captures full response, strips markdown fences, prompts `[Y/n/e]` (Y is default)
- **Edit mode**: `e` in confirm_and_run opens `$EDITOR` with the original prompt and proposed command as context (comment lines stripped on save)
- **Model cache** (`~/.cache/llm-cmd/models.json`): loaded for tab-completion, refreshed every 12h via detached subprocess (`_maybe_update_models_bg`)
- **Config** (`~/.config/llm-cmd/config.json`): persistent default model; priority: env var > config file > hardcoded fallback
- **History** (`~/.local/share/llm-cmd/history.db`): SQLite, one row per LLM call (timestamp, model, tokens, cost, mode)
- **Usage stats**: printed to stderr after each response unless `-q/--quiet` or stdout not a TTY
- **Provider config**: resolved at module level from env vars — changing provider requires no code changes
- **Entry points**: `llm-cmd`, `llm-cmd-model`, `llm-cmd-status`, `llm-cmd-cost`

Branch strategy: `main` = stable tagged releases, `dev/*` = feature branches, merge to main when tests pass.

## Documentation rule

**Every feature addition or behaviour change must update:**
1. The `_TLDR` string in `llm_cmd.py` (quick reference)
2. The `_DOCS` string in `llm_cmd.py` (man-page style)
3. `README.md` (user-facing)
4. `CLAUDE.md` Architecture section (this file)

## Git rules

- Every `git commit` is immediately followed by `git push origin <branch>` — local-only commits do not exist.
- Install with `uv tool install -e .` (never `pip install`).
