# llm-cmd

Minimal no-TUI CLI for LLMs. Ask questions or run AI-generated shell commands — **no quotes needed**.

```bash
llm-cmd what is the meaning of anagnorisis
llm-cmd -e update all my cargo binaries
llm-cmd -c write a python function that flattens a nested list
```

## Install

```bash
uv tool install -e .
```

Requires Python ≥ 3.14 and an [OpenRouter](https://openrouter.ai) API key (or any OpenAI-compatible provider).

## Tab completion

```bash
# Add to .bashrc / .zshrc
eval "$(register-python-argcomplete llm-cmd)"

# Populate model cache (auto-refreshes every 12h)
llm-cmd --update-models
```

Then `llm-cmd -m <Tab>` completes model names. Substring match: typing `haiku` finds `anthropic/claude-3-5-haiku-20241022`.

## Usage

```
llm-cmd [-e] [-c] [-m MODEL] [-s SYSTEM] [words ...]
```

| Flag | Description |
|------|-------------|
| *(none)* | Ask a question, stream the answer |
| `-e` | Generate a shell command, confirm `[y/N/e]`, run it |
| `-c` | Generate code, print to stdout |
| `-m MODEL` | Override model (default: `openai/gpt-4o-mini`) |
| `-s PROMPT` | Override system prompt |
| `--update-models` | Force-refresh model cache |
| `--list-models` | Print cached model IDs |

Stdin is also supported:

```bash
cat error.log | llm-cmd explain this error
```

## Configuration

All config via environment variables — no config file needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key (required unless `LLM_CMD_API_KEY` set) |
| `LLM_CMD_MODEL` | `openai/gpt-4o-mini` | Default model |
| `LLM_CMD_API_KEY` | `$OPENROUTER_API_KEY` | Override API key |
| `LLM_CMD_API_URL` | OpenRouter endpoint | Any OpenAI-compatible URL |

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

## Development

```bash
uv run pytest tests/ -v
uv run llm_cmd.py <prompt>   # without installing
```

## Acknowledgements

Inspired by [aichat](https://github.com/sigoden/aichat) by [@sigoden](https://github.com/sigoden) — particularly its frictionless no-quote prompt UX and `-e` execute mode. `llm-cmd` is a minimal Python reimplementation focused on fast startup and zero configuration.
