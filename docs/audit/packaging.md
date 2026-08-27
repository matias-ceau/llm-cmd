# Packaging & Release Audit — quipcli

Scope: `pyproject.toml`, `uv.lock`, CI/CD presence, git tags vs. version bumps,
entry-point flow (`__init__.py`/`__main__.py`/`entry.py`), README install
instructions vs. reality, and packaging hygiene (`.gitignore`, committed
artifacts). Read-only audit — no source files were modified. Verified live
where possible: `uv build`, `uv lock --check`, `uv run pytest -q`, and `qp
--version` were actually run against the working tree.

Audited at commit `68bff6c` (`main`, clean working tree, up to date with
`origin/main`).

## Key facts

| Item | Value |
|---|---|
| `[project.scripts]` | exactly one: `qp = "quipcli:main"` — no leftover `llm-cmd*` entries |
| `version` (pyproject.toml) | `0.4.0` |
| Latest git tag | `v0.3.0` (repo is 3 commits ahead, untagged) |
| All tags | `v0.1.0-alpha, v0.1.1, v0.1.2, v0.1.3, v0.2.0, v0.2.1, v0.3.0` — all semver-clean |
| Dependencies | `["argcomplete"]` only — claim of "zero third-party deps except argcomplete" holds |
| Dev deps | `dependency-groups.dev = ["pytest"]` — `ruff`/`pyright` not declared anywhere in the project |
| `requires-python` | `>=3.14` |
| `.python-version` | `3.14` — consistent with `requires-python` |
| Build backend | `hatchling` — `uv build` succeeds cleanly (sdist + wheel) |
| `uv lock --check` | passes, lock is in sync |
| `[tool.ruff]` / `[tool.pyright]` / `[tool.pytest]` | none present, no `ruff.toml`/`pyrightconfig.json` either |
| CI/CD | **none** — no `.github/workflows/`, only `.github/copilot-instructions.md` |
| `LICENSE` file | **absent**; no `license` field in `[project]` either |
| `[project]` authors/urls/classifiers | **absent** |
| Test suite | `uv run pytest -q` → 251 passed |
| Committed build artifacts | none (`dist/`, `build/`, `*.egg-info` not tracked) |
| `git status` | clean, `main` up to date with `origin/main` |

## Findings

### High

- **Version 0.4.0 exists in `pyproject.toml` but has no corresponding git tag.**
  `git blame` traces `version = "0.4.0"` to commit `e94b98d` ("Rename
  llm-cmd to quipcli…"), folded into the rename rather than a dedicated
  "Bump version to 0.4.0" commit (unlike the 0.2.0/0.3.0 bumps, which were
  each their own commit *and* tagged `v0.2.0`/`v0.3.0`). `git describe
  --tags` currently resolves to `v0.3.0-3-g68bff6c`. Anyone building or
  releasing from `main` right now ships "0.4.0" with no way to `git
  checkout v0.4.0` to reproduce it.
  *Fix: tag the current `main` HEAD (or the `e94b98d`/rename merge commit)
  as `v0.4.0 -a`, and going forward always pair a version bump commit with
  its tag before the next feature commits land on top.*

- **No CI at all.** There is no `.github/workflows/*.yml` (only
  `copilot-instructions.md`, which is unrelated). CLAUDE.md's "Validation
  Checklist" (ruff, pyright, pytest) is entirely manual — nothing enforces
  it on PRs or pushes, and nothing would have caught the untagged-version
  drift above automatically. Given the branch strategy (`dev/*` → `main`
  merge "when all tests pass") relies on a human/agent remembering to run
  the checklist, this is a real gap, not cosmetic.
  *Fix: add a minimal `.github/workflows/ci.yml` running `uv run ruff
  check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest
  -q` on push/PR to `main` and `dev/*`.*

### Medium

- **`ruff` and `pyright` are used but never declared as project
  dependencies.** `pyproject.toml`'s `dependency-groups.dev` only lists
  `pytest`. `uv run ruff --version` / `uv run pyright --version` currently
  succeed on this machine only because both are installed system-wide
  (`/usr/bin/ruff`, `/usr/bin/pyright`) and `uv run` falls through to PATH
  for tools it doesn't manage. On a clean machine/CI runner without those
  system packages, `uv run ruff …` / `uv run pyright` from CLAUDE.md's
  checklist would fail outright — the tools wouldn't be there. This also
  means there's no pinned/reproducible ruff or pyright version.
  *Fix: `uv add --dev ruff pyright` so both are locked in `uv.lock` and
  resolve identically everywhere.*

- **No `[tool.ruff]` / `[tool.pyright]` config sections.** Both tools run
  with 100% defaults — no line length, target-version, ignored rules, or
  strict-mode settings pinned anywhere. Given `requires-python = ">=3.14"`,
  pyright in particular should probably be told the target Python version
  explicitly rather than relying on environment auto-detection.
  *Fix: add `[tool.ruff]` (with `target-version = "py314"`) and
  `[tool.pyright]` sections to `pyproject.toml`, matching what CLAUDE.md
  already prescribes as the required tooling.*

- **`.gitignore` is minimal** (`__pycache__/`, `*.pyc`, `.venv/`,
  `*.egg-info/` only — 4 lines). It's not currently causing any problem
  (verified: no `dist/`, `build/`, cache, or data files are tracked in
  git), but it's missing common entries that would silently start getting
  picked up by a future `git add -A`/`git add .`: `dist/`, `build/`,
  `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.uv-cache/`. Low risk
  today only because CLAUDE.md's global rules explicitly forbid `git add
  -A`/`git add .` — this is a safety net for if that rule is ever missed.
  *Fix: extend `.gitignore` with the standard Python/uv/tooling cache
  entries.*

- **No `LICENSE` file and no `license` field in `pyproject.toml`.** The
  project is on GitHub with no stated license, which legally defaults to
  "all rights reserved" — anyone finding the repo can't safely reuse it,
  and `uv tool install -e .` / `pip`-style tooling has no license metadata
  to surface. This may be intentional for a personal tool, but worth a
  deliberate decision rather than an omission.
  *Fix: either add a `LICENSE` file + `license = {text = "..."}` (or
  SPDX `license = "MIT"` etc.) if it should be open, or explicitly note
  "unlicensed/personal use" somewhere if not.*

### Low

- **`[project]` metadata is thin**: no `authors`, no `[project.urls]`
  (e.g. `Repository = "https://github.com/matias-ceau/quipcli"`), no
  `classifiers`. Doesn't block `uv tool install -e .` from a local clone
  (verified working), but matters if the project is ever published to
  PyPI or if someone runs `pip show quipcli` / looks at package metadata
  for provenance.
  *Fix: add `authors = [{name = "Matias Ceau", email = "..."}]` and
  `[project.urls]` with the GitHub repo link.*

- **README install flow verified accurate.** `uv tool install -e .` →
  `qp = "quipcli:main"` entry point → `qp --version` all check out
  end-to-end (`qp 0.4.0`, already installed at `/home/matias/.local/bin/qp`
  on this machine). `register-python-argcomplete qp` matches the single
  `qp` binary. No action needed; noted only as confirmation this part of
  the audit passed.

- **Entry-point/import chain is clean.** `quipcli/__init__.py` re-exports
  the full public API (facade pattern) ending in `main` from `entry.py`;
  `quipcli/__main__.py` is a 3-line `from .entry import main; main()`, so
  `python -m quipcli` works correctly. No circular imports detected in the
  `__init__.py` import order (constants → config → context → db → models →
  multimodal → http_client → execute → tools → agent → cli → tui →
  entry — each only depends on modules already imported above it).
  `constants.py`'s `_migrate_legacy_data()` runs unconditionally at import
  time, but it's wrapped in a bare `try/except OSError: pass` around all
  filesystem calls, so a read-only or broken filesystem degrades silently
  (migration just doesn't happen) rather than raising ImportError. This is
  intentionally permissive — worth knowing it's there, not a bug.

- **`requires-python = ">=3.14"` is aggressive** (3.14 shipped ~Oct 2025).
  Not a defect — `uv` transparently downloads/manages 3.14 for users who
  don't have it system-wide — but it does mean anyone using a bare `pip
  install` or an older system Python is locked out entirely. Given
  CLAUDE.md mandates `uv` exclusively for this project, this is likely
  intentional; flagging only as a note for future contributors coming from
  a non-`uv` workflow.

## Summary of what's *not* broken

- Single clean entry point, no legacy `llm-cmd-*` executables anywhere.
- `uv.lock` is in sync with `pyproject.toml` (`uv lock --check` passes).
- `uv build` produces a valid sdist + wheel with zero warnings.
- No stray `dist/`, `build/`, `.egg-info`, or cache/data files committed to git.
- 251/251 tests pass on the current `main`.
- README's documented install/completion/model-cache flow matches the
  actual code.
