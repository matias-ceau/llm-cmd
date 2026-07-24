_TLDR = """\
qp — quick reference

  qp what is anagnorisis               free-text question (no quotes needed)
  git diff | qp summarize this         piped stdin is appended to the prompt
  qp describe this photo.jpg           multimodal: auto-detect files in words
  qp -i photo.jpg what is this         multimodal: explicit file input
  qp -e update all cargo bins          generate + confirm + run a shell command
  qp -c write a merge sort             generate code to stdout
  qp -m anthropic/claude-3 ...         use a specific model
  qp -q ...                            suppress usage stats
  NO_COLOR=1 qp ...                    disable ANSI markdown styling
  qp -s myconv ask something           start or continue a named session
  qp -s auto ask something             auto-named session (name shown on stderr)
  qp -f what did I just ask            follow-up on last session

  qp --models                          list cached models (* = default)
  qp --models --in image               models accepting image input
  qp --models --out audio              models that can generate audio
  qp --model-get                       print current default model
  qp --model-set openai/gpt-4o         set default model (saved to config)
  qp --model-set haiku                 set by unique substring match
  qp --model-set                       pick a model (fzf if available, else a numbered list)
  qp --config-edit                     open ~/.config/quipcli/config.json in $EDITOR
                                        (add "system_prompt" for standing instructions)
  qp --tui                             interactive fzf picker for models + config
  qp --status                          show current configuration + machine context
  qp --cost [1d|7d|30d|all]            show cost summary (default 7d)

  qp --update-models                   refresh model cache from provider
  qp --version                         show version
  qp --tldr                            this cheatsheet
  qp --docs                            full documentation

  Offline? If the provider is unreachable (or no API key is set), qp
  falls back to a local Ollama (config "ollama_model" or first available).\
"""

_DOCS = """\
qp(1)                               User Commands                               qp(1)

NAME
    qp — minimal CLI for LLMs, with an optional fzf-based --tui

SYNOPSIS
    qp [-e|-c] [-m MODEL] [-S SYSTEM] [-s SESSION|-f] [-i FILE] [-q] [words ...]
    qp --update-models | --models [--in MOD] [--out MOD] | --model-get | --model-set [MODEL]
    qp --config-edit | --status | --cost [PERIOD] | --tui
    qp --tldr | --docs | --version

DESCRIPTION
    qp sends a prompt to an LLM and streams the response to stdout.
    No quoting is needed — words on the command line are joined into the prompt.
    Files (images, PDFs, audio, video) are auto-detected by extension in the
    word list, or supplied explicitly with -i.
    In chat mode on TTY output, markdown gets lightweight ANSI styling
    (headings, inline/fenced code, bold, list items, blockquotes) while
    preserving streaming behavior.

    Everything is a single binary controlled by flags — there are no
    separate subcommand executables.

OPTIONS
    words               Prompt words, joined with spaces. Files detected by
                        extension are passed as multimodal content. Piped stdin
                        is appended after the words (blank-line separated), so
                        `git diff | qp summarize this` sends both; with no
                        words at all, stdin becomes the whole prompt.

    -e, --execute       Execute mode: generate a shell command, confirm, then run.
                        Prompts [Y/n/e] — Y is default (Enter to confirm).
                        Press e to open the command in $EDITOR with context.

    -c, --code          Code mode: generate raw code to stdout (no prose).

    -m, --model MODEL   Model to use for this invocation. MODEL may be a substring
                        that uniquely matches a cached model id (e.g. "haiku");
                        the resolved id is printed to stderr.
                        Default: $LLM_CMD_MODEL, config file, or openai/gpt-4o-mini.

    -S, --system PROMPT Fully override the system prompt (skips machine context
                        and config "system_prompt" injection — use this for a
                        one-off, exact system message).

    -s, --session NAME  Attach to a named session. Use 'auto' to generate a
                        timestamped name (printed to stderr for reuse).

    -f, --follow-up     Continue the last session in history. Mutually exclusive
                        with --session. The session name is printed to stderr.

    -i, --input FILE    Explicitly pass a file as multimodal input. Repeatable.
                        Files are also auto-detected from words by extension.

    -q, --quiet         Suppress post-response usage stats (model, tokens, cost)
                        and the informational Model:/Session: stderr lines.

    --update-models     Fetch and cache the model list from the provider, then exit.

    --models            List cached models (* marks the current default), then exit.
      --in MOD,MOD       Filter by required input modalities (comma-separated).
      --out MOD,MOD       Filter by required output modalities.
                        Valid modalities: text, image, audio, video, file.

    --model-get         Print the current default model, then exit.

    --model-set [MODEL] Set the default model (written to config file), then exit.
                        MODEL may be a substring that uniquely matches a cached
                        model id. If omitted: launches the fzf model picker
                        (see --tui) when fzf is installed, otherwise falls back
                        to a numbered list read from stdin.

    --config-edit       Open the config file directly in $EDITOR, then exit.

    --status            Print current configuration: API URL, key (masked),
                        model, paths to config/cache/history, then exit.

    --cost [PERIOD]     Show usage cost summary, then exit.
                        PERIOD: 1d, 7d (default), 30d, or all.

    --tui               Interactive fzf-based picker for models and config
                        (see TUI below). Requires fzf; bat is used for preview
                        syntax highlighting when available.

    --version           Print version and exit.

    --tldr              Show quick-reference cheatsheet and exit.

    --docs              Show this documentation and exit.

MULTIMODAL
    Supported input formats:
      Images : .jpg .jpeg .png .gif .webp  (also accepts https:// image URLs)
      PDFs   : .pdf
      Audio  : .mp3 .wav .ogg
      Video  : .mp4 .webm

    If the selected model does not support the required input modality, qp
    prints an error and lists compatible models from the cache.

SESSIONS
    Sessions group messages into multi-turn conversations. Each exchange
    (without -s/-f) is a standalone interaction stored in history.

    qp -s myproject explain the architecture
    qp -s myproject what about the tests ?
    qp -f any other suggestions ?   # continues last session

TUI (--tui)
    Interactive picker built on the real fzf/bat binaries (not a
    reimplementation) — same conventions as fzf's own FZF_DEFAULT_OPTS if
    you already have one configured.

    Top menu: Models / Config.

      Models   Fuzzy list of cached models, current default marked with *.
               Right-hand preview: name, context length, price per 1M
               tokens (prompt/completion), input/output modalities.
               Enter sets the highlighted model as the new default.
               ctrl-r refreshes the cache from the provider in place.

      Config   Fuzzy list of editable keys (default_model, system_prompt,
               ollama_model). Right-hand preview shows the live config.json
               via bat. Enter on default_model drills into the Models list;
               on ollama_model, into a list of locally available Ollama
               models (falls back to free-text entry if Ollama is
               unreachable); on system_prompt, opens $EDITOR on just that
               value. ctrl-e opens the whole config file in $EDITOR at any
               time. Esc returns to the previous list; Esc at the top menu
               exits.

OLLAMA FALLBACK
    When the provider is unreachable (after retries) or no API key is set,
    qp falls back to a local Ollama instance if one is running.
    The model is taken from the "ollama_model" config key, or the first
    locally available model otherwise. http:// endpoints (e.g. a permanent
    LLM_CMD_API_URL pointing at Ollama) never require an API key.
    Transient provider errors (429/5xx) are retried with backoff first;
    authentication errors (401…) do NOT trigger the fallback.

ENVIRONMENT
    LLM_CMD_MODEL       Default model name.
    LLM_CMD_API_KEY     API key (takes priority over OPENROUTER_API_KEY).
    LLM_CMD_API_URL     Full endpoint URL (default: OpenRouter).
    LLM_CMD_OLLAMA_URL  Local Ollama base URL (default: http://localhost:11434).
    OPENROUTER_API_KEY  OpenRouter API key (fallback).
    NO_COLOR            Disable ANSI markdown styling in streamed chat output.
    XDG_CACHE_HOME      Cache directory (default: ~/.cache).
    XDG_CONFIG_HOME     Config directory (default: ~/.config).
    XDG_DATA_HOME       Data directory (default: ~/.local/share).
    EDITOR              Editor for -e edit mode, --config-edit, and --tui
                        text fields (default: vi).
    SHELL               Shell name used in execute-mode system prompt.

CONTEXT INJECTION
    Unless -S/--system fully overrides it, every request's system prompt is
    built from up to three parts, in order:
      1. mode-specific instructions (execute/code mode only)
      2. machine context — OS/distro, $SHELL, architecture — computed fresh
         on every call, so the same config.json works correctly across
         different machines without editing it per host
      3. the "system_prompt" key from config.json, if set — free-text
         instructions/preferences that should apply to every call (e.g.
         "prefer pacman over apt-get", "I use zsh and neovim")

    Set persistent instructions with:
        qp --config-edit          # add "system_prompt": "..." to config.json
        qp --tui                  # or interactively, via Config > system_prompt

FILES
    ~/.config/quipcli/config.json       Persistent config: default_model,
                                         system_prompt, ollama_model.
                                         Auto-created on first run; edit it
                                         directly or via `qp --config-edit`.
    ~/.cache/quipcli/models.json        Cached model list (12h TTL).
    ~/.local/share/quipcli/history.db   Usage history + sessions (SQLite).

    On first run, if the above don't exist yet but a pre-rename
    ~/.config/llm-cmd (etc.) layout does, its config/cache/history are
    copied over automatically (one-time, non-destructive).

USAGE STATS
    After each response, qp prints to stderr:
        [model | N tok | $0.0012]
    Suppressed with -q/--quiet or when stdout is not a TTY.\
"""
