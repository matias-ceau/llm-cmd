# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-cmd` is a minimal no-TUI Python CLI for LLMs. Core use cases:
- `llm-cmd what is the meaning of anagnorisis` — free-text prompt, no quotes
- `llm-cmd -e update all my cargo binaries` — generate + confirm + run a shell command

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
- **Streaming** (`call_llm_streaming`): SSE parsed line-by-line, tokens printed as received
- **Execute mode** (`confirm_and_run`): captures full response, strips markdown fences, prompts `[y/N/e]`
- **Model cache** (`~/.cache/llm-cmd/models.json`): loaded for tab-completion, refreshed every 12h via detached subprocess (`_maybe_update_models_bg`)
- **Provider config**: resolved at module level from env vars — changing provider requires no code changes

Branch strategy: `main` = stable tagged releases, `dev/*` = feature branches, merge to main when tests pass.
