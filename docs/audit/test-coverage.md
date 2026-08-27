# quipcli test-coverage audit

Date: 2026-08-27 · Read-only audit, no source/test changes made.

## Methodology

- Static mapping: `tests/test_quipcli.py` (single file, 246 `test_` functions across 38
  `Test*` classes) read via class/def names and imports; test bodies skimmed to judge
  depth, not read line-by-line.
- Dynamic: ran the suite for real.
  - `uv run pytest tests/ -q` → **251 passed** (246 top-level + parametrized variants),
    0 failed, 0.55s. Suite passes cleanly.
  - `pytest-cov` is **not** a declared dev dependency (`pyproject.toml` `[dependency-groups] dev = ["pytest"]`
    only). Installed it ephemerally via `uv run --with pytest-cov` (no project files touched)
    to get a real line-coverage report: `uv run --with pytest-cov pytest tests/ --cov=quipcli --cov-report=term-missing -q`.
  - Result: **82% overall line coverage** (1381 stmts, 251 missed). Per-module numbers and
    missing line ranges below are from that real run, not guessed.
- Gap analysis below cross-references the coverage report against the specific risk areas
  named in the task (Ollama fallback, retry/backoff, SSE streaming, agent loop, tui,
  legacy migration, config precedence, cost aggregation, multimodal edge cases, model-name
  resolution, markdown ANSI rendering).

## Module-by-module table

| Module | Test file/class(es) | Line cov | Depth | Notable gaps |
|---|---|---|---|---|
| `constants.py` | (indirect only, no `TestConstants`) | 91% | Shallow | `_migrate_legacy_data`'s actual copy branch + `except OSError` never exercised (see High #1) |
| `config.py` | `TestConfig`, `TestSeedDefaults` | 100% | Deep | env>config>hardcoded precedence, `_ensure_config` idempotency, `_seed_defaults` no-overwrite all covered |
| `context.py` | `TestMachineContext` | 84% | Shallow | `_linux_distro` OSError-on-read branch untested (only "file exists" and "file missing" paths) |
| `db.py` | `TestHistory`, `TestSessions` | 92% | Moderate | every DB function's broad `except Exception: pass/return {}` fallback (lines 89-90, 100-101, 124-125, 177-178) untested — no test simulates a corrupt/locked DB |
| `models.py` | `TestModelsUrl`, `TestLoadModels`, `TestResolveModelName`, `TestMaybeUpdateModelsBg`, `TestFetchRankings`, `TestRankingFor` | 76% | Uneven | **`_fetch_models` (lines 63-84) has zero tests** despite `_fetch_rankings`, its structural twin, having 5 (High #2). `_resolve_model_name` ambiguous/no-match/empty-cache paths are well covered. Truncation-at-10 output branches (`_check_modality_support`, `_resolve_model_name`) untested (Low) |
| `multimodal.py` | `TestIsImageUrl`, `TestBuildUserContent`, `TestEncodeFileContent`, `TestModalitySupport` | 90% | Moderate | oversized-file warning branch (>`_MAX_FILE_BYTES`) untested; `_build_user_content` end-to-end only exercised with images — PDF/audio/video modality-detection (`detected.add(...)`) only tested via direct `_encode_file_content` calls, not through the full content-builder path |
| `http_client.py` | `TestMakeRequest`, `TestOllamaFallback`, `TestCallLlmStreaming`, `TestCallLlmCapture`, `TestMakeRequestExtra` | 90% | Good on retry/fallback, gap on SSE chunking | See High #3 (streaming carry-buffer) and Medium #1 (retry status set) below |
| `tools.py` | `TestToolsModule` | 94% | Deep | confirm/decline for both `run_shell` and `write_file`, Ctrl-C-as-decline, timeout, truncation, unknown-tool/invalid-JSON dispatch — all covered. Minor: 2 small untested lines (65-66, 158-159), low risk |
| `agent.py` | `TestAgentLoop` | 90% | Good on control flow, gap on 2 branches | final-answer, tool-exec round-trip, server-tool passthrough, max-steps cutoff, `--no-web`, error payload all covered (see Medium #2 for what's left) |
| `execute.py` | `TestConfirmAndRun`, `TestEditInEditor` | 84% | Uneven | `confirm_and_run` (y/n/e/Ctrl-C/fence-stripping) and `_edit_in_editor` well tested; **`_edit_text_value` (lines 41-49) is never tested directly** — every call site mocks it out (Medium #3) |
| `cli.py` | `TestParser`, `TestGetContent`, `TestExecutePrompt` | 82% | Good on parser/stdin, gap on stats | argparse flags and `get_content`'s stdin/words/media combos are thorough; **`_print_stats` (lines 215-224) has no direct test** — only reachable today via untested `main()` |
| `tui.py` | `TestTuiHelpers`, `TestTuiModelInfo`, `TestModelsView`, `TestConfigView`, `TestRunTui`, `TestRunFzf` | 97% | Deep | Best-covered risky module in the codebase — achieved by mocking `_run_fzf`/`subprocess`/`shutil.which` rather than shelling out to real `fzf`/`bat`. This is a deliberate, working strategy, not "untested by design." Minor misses: `--ansi`/`--border-label` arg-building branches in `_run_fzf`, `FZF_PREVIEW_COLUMNS` invalid-int fallback |
| `docs.py` | none | 0% | None | `_TLDR`/`_DOCS` are static strings; only reachable via untested `main()` `--tldr`/`--docs` branches. Low risk (no logic) but genuinely 0% |
| `entry.py` | `TestModePrompt`, `TestMainStatus`, `TestModelConfigFlags` (`_do_status`/`_do_cost`/`_do_model_*`/`_do_config_edit` tested as standalone functions) | **47%** | Shallow — `main()` itself is untested | **`main()` (lines 165-318, ~150 lines) has zero direct test coverage** — see High #1 in prioritized list. All its helpers (`_mode_prompt`, `_do_status`, `_do_cost`, model/config flag handlers) are tested in isolation, but the orchestration function that wires argv → mode dispatch → LLM call → stats/history recording is never invoked by the suite |
| `__init__.py` | (facade, exercised transitively by every `quipcli.X` test call) | 100% | N/A | re-export shim, fully covered by usage |
| `__main__.py` | none | 0% | None | 2-line `python -m quipcli` shim calling `main()`; trivial, low risk |

## Prioritized gaps

### High

1. **`entry.py::main()` is completely untested (lines 165-318, 47% module coverage).**
   No test invokes `main()` end-to-end. Everything main() alone does — hidden `--_tui-*`
   flags, `--tldr`/`--docs`/`--update-models`/`--update-rankings` early returns, the
   `_resolve_model_name` "Model: X" stderr notice, session/follow-up wiring into
   `get_content`, the execute/agent/code/chat mode dispatch (`if args.execute: ... elif
   args.agent: ... elif args.code: ... else:`), and `_post()`'s stats+history recording —
   is invisible to the suite. This is the actual product wiring; a bug here (e.g. wrong
   mode routed to `call_llm_capture` vs `call_llm_streaming`, `_post` called with wrong
   args, quiet-flag suppression broken) would silently ship.
   *Suggest*: a handful of `main()`-level integration tests, one per mode (`-e`/`-a`/`-c`/
   chat default), driving `sys.argv` + mocked `_make_request`/`http.client.HTTPSConnection`,
   asserting the right `call_llm_*` function was invoked with the right messages/model and
   that `_record_usage`/`_record_message` got called. Also one test for the "Model: X"
   resolved-name stderr notice and one for `-q` suppressing it.

2. **`models.py::_fetch_models` has zero tests** (lines 63-84, the function behind
   `qp --update-models` and the 12h background refresh). Its structural twin,
   `_fetch_rankings`, has 5 tests covering success/non-openrouter-skip/no-key-skip/
   non-200/invalid-JSON — none of that exists for `_fetch_models`.
   *Suggest*: mirror `TestFetchRankings` as `TestFetchModels`: success caches sorted IDs
   to `_MODELS_CACHE` via `_atomic_write_text`, connection `OSError` returns `[]` +
   stderr message, non-200 status returns `[]`, invalid JSON returns `[]`.

3. **Streaming SSE carry-buffer logic is never exercised with a split/partial line**
   (`http_client.py` `_MarkdownAnsiRenderer.render()` lines 311-321, `finish()` lines
   323-337). Every `TestCallLlmStreaming` test puts a full multi-line `delta.content`
   string, already newline-terminated, into a single SSE `data:` chunk. Real streaming
   sends many small token-level deltas with no trailing newline, which is exactly what
   `render()`'s `self._carry = text; return ""` branch and `finish()`'s "flush unterminated
   state at end of stream" branch exist to handle — and both are 0% covered. This is a
   plausible source of visible bugs (dropped/garbled output, stuck ANSI codes) that
   wouldn't show up until real traffic.
   *Suggest*: a test feeding content split into several small chunks with no embedded
   newlines (e.g. `"**bo"`, `"ld** end\n"`) and asserting the final rendered text/ANSI
   state matches feeding it as one chunk; a test where the stream ends mid-`**bold**`/
   mid-code-fence (no closing marker before `[DONE]`) asserting `finish()` still emits a
   trailing RESET and doesn't leave the terminal in a colored state.

### Medium

1. **Retry logic only exercises one of five retriable statuses.** `_RETRIABLE_STATUSES =
   {429, 500, 502, 503, 529}` (http_client.py:12) share one code path, and only 429 is
   tested (`test_retries_on_429_then_succeeds`). Lower risk since it's a single `if status
   in _RETRIABLE_STATUSES` check, but there's also no test for **retry exhaustion on a
   persistently-failing transient status** (does it fall through to Ollama fallback, or
   exit with the last error?) — only connection-error exhaustion is tested
   (`test_connection_error_no_ollama_exits`), not HTTP-status exhaustion.
   *Suggest*: one parametrized test over `{500, 502, 503, 529}` reusing the 429 fixture;
   one test where all 3 retries return 429/503 and it still fails, asserting the final
   exit/error path.

2. **`agent.py` two untested branches**: malformed JSON from the API mid-loop (lines
   42-44, `json.JSONDecodeError` → exit) and `_print_final` with markdown rendering
   actually enabled (lines 97-100) — every `TestAgentLoop` test passes
   `render_markdown=False`, so the agent's own markdown-rendering path is unverified
   (only the top-level `call_llm_streaming` markdown renderer is tested).
   *Suggest*: one test with invalid JSON body → `SystemExit(1)`; one test with
   `render_markdown=True` + `_use_markdown_rendering` patched True asserting ANSI codes
   appear in the final answer.

3. **`execute.py::_edit_text_value` has no direct test** — every caller in `tui.py`
   mocks it out. Its sibling `_edit_in_editor` (same tempfile+`$EDITOR`+cleanup pattern,
   used for `-e`'s command editing) has two direct tests
   (`test_strips_comment_lines`, `test_editor_with_flags_split_correctly`). Low risk of
   divergence since the code is nearly identical, but it's the one code path in the
   module the suite never actually runs.
   *Suggest*: port `TestEditInEditor`'s two tests to `_edit_text_value` (no comment-header
   assertions needed, just tempfile write → fake `$EDITOR` mutates it → stripped read-back).

4. **`cli.py::_print_stats` has no direct test.** Formats/prints the `[model | N tok |
   $cost]` stderr line after every response; currently only reachable through the
   untested `main()`. Cheap to test in isolation.
   *Suggest*: 2-3 tests: `stats=None` prints nothing; stats with tokens+cost formats both;
   stats with `cost_usd=None` omits the `$` segment.

5. **`db.py`'s broad `except Exception: pass` / `return {}` fallbacks are all untested**
   (`_record_message`, `_get_session_messages`, `_last_session_id`, `_cost_summary` —
   4 call sites). These exist specifically to make history-recording failures non-fatal,
   but nothing verifies that a corrupt/locked DB actually degrades gracefully rather than
   masking a real bug.
   *Suggest*: at least one test per function that forces the `except` (e.g. monkeypatch
   `_db_conn` to raise, or point `_HISTORY_DB` at a non-DB file) and asserts the
   documented graceful-degradation behavior (silent no-op / empty list / empty dict).

### Low

- `constants.py::_migrate_legacy_data`: only the "old files absent" no-op path runs
  (implicitly, at import time). No test creates a legacy `~/.config/llm-cmd/config.json`
  etc. and asserts it gets copied to the new XDG location, nor a test that a copy
  `OSError` (e.g. permission denied) is swallowed rather than crashing startup. Grouped
  as Low rather than Medium because it's one-shot, idempotent, and copy-only (no data
  loss risk on failure) — but it is explicitly named in the task and currently has zero
  coverage (`grep -n "migrate_legacy\|llm-cmd" tests/test_quipcli.py` → no hits).
  *Suggest*: one test with a populated legacy dir + fresh XDG dirs asserting the copy
  happens and old files are untouched; one test where the destination already exists
  asserting no overwrite; one where `shutil.copy2` raises `OSError` asserting no crash.
- `models.py`/`context.py`: ">10 matches" truncation-message branches
  (`_resolve_model_name`, `_check_modality_support`), `_linux_distro`'s `OSError` read
  branch — cosmetic output formatting, low blast radius.
- `tui.py::_run_fzf`: `--ansi` and `--border-label` argv-building branches, and the
  `FZF_PREVIEW_COLUMNS` invalid-int fallback in `_print_model_info` — cosmetic, tui.py is
  otherwise the best-tested module (97%) via a solid subprocess-mocking strategy already
  in place worth reusing for these.
- `multimodal.py`: oversized-file (`> _MAX_FILE_BYTES`) warning-print branch untested.
- `docs.py`, `__main__.py`: 0% coverage but no branching logic (static strings / 2-line
  shim) — not worth dedicated tests; would be covered incidentally once `main()` gets
  integration tests (High #1).
- `pytest-cov` is not in `pyproject.toml`'s `dev` group, so `--cov` reports aren't
  reproducible without the `uv run --with pytest-cov` workaround used for this audit.
  Not a code gap, but worth adding (`uv add --dev pytest-cov`) if coverage tracking is
  meant to be routine.
