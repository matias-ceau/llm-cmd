_TLDR = """\
llm-cmd — quick reference

  llm-cmd what is anagnorisis          free-text question (no quotes needed)
  llm-cmd describe this photo.jpg      multimodal: auto-detect files in words
  llm-cmd -i photo.jpg what is this    multimodal: explicit file input
  llm-cmd -e update all cargo bins     generate + confirm + run a shell command
  llm-cmd -c write a merge sort        generate code to stdout
  llm-cmd -m anthropic/claude-3 ...    use a specific model
  llm-cmd -q ...                       suppress usage stats
  NO_COLOR=1 llm-cmd ...               disable ANSI markdown styling
  llm-cmd -s myconv ask something      start or continue a named session
  llm-cmd -s auto ask something        auto-named session (name shown on stderr)
  llm-cmd -f what did I just ask       follow-up on last session

  llm-cmd-model list                   list cached models (* = default)
  llm-cmd-model list --in image        models accepting image input
  llm-cmd-model list --out audio       models that can generate audio
  llm-cmd-model get                    print current default model
  llm-cmd-model set openai/gpt-4o      set default model (saved to config)
  llm-cmd-model set haiku              set by unique substring match
  llm-cmd-model set                    pick a model interactively from the cache
  llm-cmd-model edit                   open ~/.config/llm-cmd/config.json in $EDITOR
  llm-cmd-status                       show current configuration
  llm-cmd-cost [--period 1d|7d|30d]    show cost summary

  llm-cmd --update-models              refresh model cache from provider
  llm-cmd --tldr                       this cheatsheet
  llm-cmd --docs                       full documentation\
"""

_DOCS = """\
llm-cmd(1)                         User Commands                        llm-cmd(1)

NAME
    llm-cmd — minimal no-TUI CLI for LLMs

SYNOPSIS
    llm-cmd [-e|-c] [-m MODEL] [-S SYSTEM] [-s SESSION|-f] [-i FILE] [-q] [words ...]
    llm-cmd --update-models | --list-models | --tldr | --docs
    llm-cmd-model list [--in MODALITIES] [--out MODALITIES]
    llm-cmd-model get | set MODEL | edit
    llm-cmd-status
    llm-cmd-cost [--period PERIOD]

DESCRIPTION
    llm-cmd sends a prompt to an LLM and streams the response to stdout.
    No quoting is needed — words on the command line are joined into the prompt.
    Files (images, PDFs, audio, video) are auto-detected by extension in the
    word list, or supplied explicitly with -i.
    In chat mode on TTY output, markdown gets lightweight ANSI styling
    (headings, inline/fenced code, bold, list items, blockquotes) while
    preserving streaming behavior.

OPTIONS
    words               Prompt words, joined with spaces. Files detected by
                        extension are passed as multimodal content. Reads stdin
                        if no words and no -i files are given.

    -e, --execute       Execute mode: generate a shell command, confirm, then run.
                        Prompts [Y/n/e] — Y is default (Enter to confirm).
                        Press e to open the command in $EDITOR with context.

    -c, --code          Code mode: generate raw code to stdout (no prose).

    -m, --model MODEL   Model to use for this invocation. MODEL may be a substring
                        that uniquely matches a cached model id (e.g. "haiku");
                        the resolved id is printed to stderr.
                        Default: $LLM_CMD_MODEL, config file, or openai/gpt-4o-mini.

    -S, --system PROMPT Override the system prompt.

    -s, --session NAME  Attach to a named session. Use 'auto' to generate a
                        timestamped name (printed to stderr for reuse).

    -f, --follow-up     Continue the last session in history. Mutually exclusive
                        with --session. The session name is printed to stderr.

    -i, --input FILE    Explicitly pass a file as multimodal input. Repeatable.
                        Files are also auto-detected from words by extension.

    -q, --quiet         Suppress post-response usage stats (model, tokens, cost).

    --update-models     Fetch and cache the model list from the provider, then exit.

    --list-models       Print cached model IDs (one per line) and exit.

    --tldr              Show quick-reference cheatsheet and exit.

    --docs              Show this documentation and exit.

SUBCOMMANDS
    llm-cmd-model list [--in MOD,MOD] [--out MOD,MOD]
                        List cached models. Current default is marked with *.
                        --in: filter by required input modalities (comma-separated).
                        --out: filter by supported output modalities.
                        Valid modalities: text, image, audio, video, file.

    llm-cmd-model get   Print the current default model.
    llm-cmd-model set [MODEL]
                        Set the default model (written to config file). MODEL
                        may be a substring that uniquely matches a cached model
                        id. If MODEL is omitted, pick interactively from a
                        numbered list of cached models.
    llm-cmd-model edit  Open the config file directly in $EDITOR.

    llm-cmd-status      Print current configuration: API URL, key, model,
                        paths to config, cache, and history database.

    llm-cmd-cost        Show usage cost summary.
      --period PERIOD   Period: 1d, 7d (default), 30d, or all.

MULTIMODAL
    Supported input formats:
      Images : .jpg .jpeg .png .gif .webp  (also accepts https:// image URLs)
      PDFs   : .pdf
      Audio  : .mp3 .wav .ogg
      Video  : .mp4 .webm

    If the selected model does not support the required input modality, llm-cmd
    prints an error and lists compatible models from the cache.

SESSIONS
    Sessions group messages into multi-turn conversations. Each exchange
    (without -s/-f) is a standalone interaction stored in history.

    llm-cmd -s myproject explain the architecture
    llm-cmd -s myproject what about the tests ?
    llm-cmd -f any other suggestions ?   # continues last session

ENVIRONMENT
    LLM_CMD_MODEL       Default model name.
    LLM_CMD_API_KEY     API key (takes priority over OPENROUTER_API_KEY).
    LLM_CMD_API_URL     Full endpoint URL (default: OpenRouter).
    OPENROUTER_API_KEY  OpenRouter API key (fallback).
    NO_COLOR            Disable ANSI markdown styling in streamed chat output.
    XDG_CACHE_HOME      Cache directory (default: ~/.cache).
    XDG_CONFIG_HOME     Config directory (default: ~/.config).
    XDG_DATA_HOME       Data directory (default: ~/.local/share).
    EDITOR              Editor for -e edit mode (default: vi).
    SHELL               Shell name used in execute-mode system prompt.

FILES
    ~/.config/llm-cmd/config.json       Persistent config (default model).
                                         Auto-created on first run; edit it
                                         directly or via `llm-cmd-model edit`.
    ~/.cache/llm-cmd/models.json        Cached model list (12h TTL).
    ~/.local/share/llm-cmd/history.db   Usage history + sessions (SQLite).

USAGE STATS
    After each response, llm-cmd prints to stderr:
        [model | N tok | $0.0012]
    Suppressed with -q/--quiet or when stdout is not a TTY.\
"""
