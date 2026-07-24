# quipcli

Minimal CLI for LLMs. Ask questions or run AI-generated shell commands — **no quotes needed**.

```bash
qp what is the meaning of anagnorisis
qp -e update all my cargo binaries
qp -c write a python function that flattens a nested list
```

Everything lives behind a single binary, `qp` — no separate subcommand executables.

## Install

```bash
uv tool install -e .
```

Requires Python ≥ 3.14 and an [OpenRouter](https://openrouter.ai) API key (or any OpenAI-compatible provider, or a local [Ollama](https://ollama.com)).

## Tab completion

```bash
# Add to .bashrc / .zshrc
eval "$(register-python-argcomplete qp)"

# Populate model cache (auto-refreshes every 12h)
qp --update-models
```

Then `qp -m <Tab>` completes model names. Substring match: typing `haiku` finds `anthropic/claude-3-5-haiku-20241022`.

## Usage

```
qp [-e] [-c] [-m MODEL] [-S SYSTEM] [-s SESSION|-f] [-i FILE] [-q] [words ...]
```

| Flag | Description |
|------|-------------|
| *(none)* | Ask a question, stream the answer |
| `-e` | Generate a shell command, confirm `[Y/n/e]`, run it |
| `-c` | Generate code, print to stdout |
| `-m MODEL` | Override model (default: `openai/gpt-4o-mini`). MODEL may be a substring matching a single cached model, e.g. `-m haiku` |
| `-S PROMPT` | Override system prompt |
| `-s SESSION` / `-f` | Named session, or follow up on the last one |
| `-q` | Quiet: suppress usage stats and informational stderr lines |
| `--models` [`--in`/`--out`] | List cached models (marked default), filterable by modality |
| `--model-get` / `--model-set [MODEL]` | Print / set the default model |
| `--config-edit` | Open `config.json` in `$EDITOR` |
| `--status` | Show current configuration |
| `--cost [PERIOD]` | Usage cost summary (`1d`/`7d`/`30d`/`all`) |
| `--tui` | Interactive fzf-based picker for models and config — see below |
| `--update-models` | Force-refresh model cache |
| `--version` | Print version |

Chat responses render lightweight ANSI markdown styling on TTYs (headings, code spans/blocks, bold, list items, blockquotes) while still streaming token-by-token. Disable colors with `NO_COLOR=1`.

Stdin is also supported, alone or combined with a prompt (piped content is appended after the words):

```bash
cat error.log | qp            # stdin as the whole prompt
git diff | qp summarize this  # words + piped context
git diff | qp -e write a conventional commit command
```

### Offline / Ollama fallback

If the provider is unreachable (after automatic retries) or no API key is set, `qp` transparently falls back to a local [Ollama](https://ollama.com) instance when one is running. The local model comes from the `"ollama_model"` config key, or the first available model otherwise. Override the Ollama URL with `LLM_CMD_OLLAMA_URL` (default `http://localhost:11434`).

## Interactive TUI (`--tui`)

```bash
qp --tui
```

Shells out to the real `fzf` (and `bat`, for preview syntax highlighting) — no reimplemented fuzzy-finder, and it inherits your existing `FZF_DEFAULT_OPTS` if you have one.

- **Models** — fuzzy list of cached models, current default marked `*`. The preview pane shows name, context length, price per 1M tokens (prompt/completion), and input/output modalities. `Enter` sets the highlighted model as the new default; `ctrl-r` refreshes the cache from the provider without leaving the picker.
- **Config** — fuzzy list of editable keys (`default_model`, `system_prompt`, `ollama_model`), with the live `config.json` shown via `bat` on the right. `Enter` on `default_model` drills into the Models list; on `ollama_model`, into a list of locally available Ollama models (falls back to free-text entry if Ollama is unreachable); on `system_prompt`, opens `$EDITOR` on just that value. `ctrl-e` opens the whole config file in `$EDITOR` at any time.

`Esc` steps back one level; `Esc` at the top menu exits. Requires `fzf` to be installed — `qp --model-set` (no value) also uses this picker when `fzf` is available, falling back to a plain numbered prompt otherwise.

## Configuration

`qp` keeps a persistent config file at `~/.config/quipcli/config.json`, created automatically on first run. Edit it in place with any text editor, or interactively:

```bash
qp --model-set            # pick a default model interactively (fzf, or a numbered list)
qp --model-set haiku      # set by substring match
qp --config-edit          # open config.json in $EDITOR
qp --tui                  # or browse/edit everything interactively
```

Environment variables always take priority over the config file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key (required unless `LLM_CMD_API_KEY` set, or the provider is a local `http://` endpoint) |
| `LLM_CMD_MODEL` | config file, or `openai/gpt-4o-mini` | Default model |
| `LLM_CMD_API_KEY` | `$OPENROUTER_API_KEY` | Override API key |
| `LLM_CMD_API_URL` | OpenRouter endpoint | Any OpenAI-compatible URL (http:// endpoints need no key) |
| `LLM_CMD_OLLAMA_URL` | `http://localhost:11434` | Local Ollama used as offline fallback |

### Persistent instructions + machine context

Every request (unless `-S` fully overrides the system prompt) automatically gets:

1. mode-specific instructions (execute/code mode)
2. **machine context** — OS/distro, `$SHELL`, architecture — detected fresh on every call, so the same `config.json` is correct whether it's synced to an Arch box or a Mac. No more reminding the model what OS you're on.
3. a free-text `"system_prompt"` from `config.json`, if you set one — standing instructions/preferences applied to every call:

```bash
qp --config-edit
# add to config.json:
# { "default_model": "...", "system_prompt": "Prefer pacman over apt-get. I use zsh and neovim." }
```

### Provider examples

```bash
# Groq (fast)
export LLM_CMD_API_URL=https://api.groq.com/openai/v1/chat/completions
export LLM_CMD_API_KEY=$GROQ_API_KEY
export LLM_CMD_MODEL=llama-3.3-70b-versatile

# Direct OpenAI
export LLM_CMD_API_URL=https://api.openai.com/v1/chat/completions
export LLM_CMD_API_KEY=$OPENAI_API_KEY
export LLM_CMD_MODEL=gpt-4o-mini
```

## Migrating from `llm-cmd`

This project was renamed from `llm-cmd` to `quipcli` (command `qp`) — the old name collided with an unrelated PyPI package. On first run, if `~/.config/quipcli` (etc.) doesn't exist yet but a pre-rename `~/.config/llm-cmd` layout does, `config.json`, `models.json`, and `history.db` are copied over automatically (one-time, non-destructive — the old files are left untouched).

## Development

```bash
uv run pytest tests/ -v
uv run python -m quipcli <prompt>   # without installing
```

## Acknowledgements

Inspired by [aichat](https://github.com/sigoden/aichat) by [@sigoden](https://github.com/sigoden) — particularly its frictionless no-quote prompt UX and `-e` execute mode. `quipcli` is a minimal Python reimplementation focused on fast startup and zero configuration.
