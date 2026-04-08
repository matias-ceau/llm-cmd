import json
import os
import re
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import llm_cmd
from llm_cmd import (
    _execute_prompt,
    _load_models,
    _maybe_update_models_bg,
    _models_url,
    _strip_fences,
    build_parser,
    get_prompt,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class MockHTTPResponse:
    """Minimal http.client.HTTPResponse stand-in."""

    def __init__(self, status: int, body: bytes, sse_lines: list[bytes] | None = None):
        self.status = status
        self._body = body
        self._lines = sse_lines or []
        self._idx = 0

    def read(self) -> bytes:
        return self._body

    def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line


def _mock_conn(resp: MockHTTPResponse) -> MagicMock:
    conn = MagicMock()
    conn.getresponse.return_value = resp
    return conn


@pytest.fixture
def mock_http():
    """Patches HTTPSConnection + API key. Yields a factory: resp -> http_cls mock."""
    with patch("http.client.HTTPSConnection") as http_cls, \
         patch("llm_cmd._API_KEY", "key"):
        def setup(resp: MockHTTPResponse) -> MagicMock:
            http_cls.return_value = _mock_conn(resp)
            return http_cls
        yield setup


# ── _strip_fences ─────────────────────────────────────────────────────────────

class TestStripFences:
    def test_plain_command_unchanged(self):
        assert _strip_fences("ls -la") == "ls -la"

    def test_bash_fence(self):
        assert _strip_fences("```bash\nls -la\n```") == "ls -la"

    def test_sh_fence(self):
        assert _strip_fences("```sh\nls -la\n```") == "ls -la"

    def test_generic_fence(self):
        assert _strip_fences("```\nls -la\n```") == "ls -la"

    def test_no_trailing_fence(self):
        assert _strip_fences("```bash\nls -la") == "ls -la"

    def test_whitespace_stripped(self):
        assert _strip_fences("  ls -la  ") == "ls -la"

    def test_multiline_content_preserved(self):
        assert _strip_fences("```bash\ncmd1 && cmd2\n```") == "cmd1 && cmd2"


# ── build_parser / get_prompt ─────────────────────────────────────────────────

class TestParser:
    def test_words_joined(self):
        args = build_parser().parse_args(["what", "is", "X"])
        assert get_prompt(args) == "what is X"

    def test_default_model(self):
        args = build_parser().parse_args(["hi"])
        assert args.model == llm_cmd.DEFAULT_MODEL

    def test_model_flag(self):
        args = build_parser().parse_args(["-m", "anthropic/claude-3-5-haiku", "hi"])
        assert args.model == "anthropic/claude-3-5-haiku"

    def test_execute_default_false(self):
        assert build_parser().parse_args(["hi"]).execute is False

    def test_execute_flag(self):
        assert build_parser().parse_args(["-e", "do it"]).execute is True

    def test_code_default_false(self):
        assert build_parser().parse_args(["hi"]).code is False

    def test_code_flag(self):
        assert build_parser().parse_args(["-c", "write a sort"]).code is True

    def test_system_flag(self):
        args = build_parser().parse_args(["-s", "be terse", "hi"])
        assert args.system == "be terse"

    def test_update_models_flag(self):
        assert build_parser().parse_args(["--update-models"]).update_models is True

    def test_list_models_flag(self):
        assert build_parser().parse_args(["--list-models"]).list_models is True


class TestGetPrompt:
    def test_stdin_fallback(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = "  piped prompt\n"
        with patch("sys.stdin", stdin):
            assert get_prompt(args) == "piped prompt"

    def test_tty_no_words_exits(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = True
        with patch("sys.stdin", stdin):
            with pytest.raises(SystemExit) as exc:
                get_prompt(args)
        assert exc.value.code == 1

    def test_empty_stdin_exits(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = "   "
        with patch("sys.stdin", stdin):
            with pytest.raises(SystemExit) as exc:
                get_prompt(args)
        assert exc.value.code == 1


# ── _execute_prompt ───────────────────────────────────────────────────────────

class TestExecutePrompt:
    def test_includes_shell_name(self, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert "zsh" in _execute_prompt()

    def test_fallback_to_bash(self, monkeypatch):
        monkeypatch.delenv("SHELL", raising=False)
        assert "bash" in _execute_prompt()

    def test_no_markdown_instruction(self, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/bash")
        prompt = _execute_prompt()
        assert "No markdown" in prompt
        assert "No code fences" in prompt


# ── _models_url ───────────────────────────────────────────────────────────────

class TestModelsUrl:
    @pytest.mark.parametrize("api_url,expected", [
        (
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/models",
        ),
        (
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/models",
        ),
        (
            "https://api.groq.com/openai/v1/chat/completions",
            "https://api.groq.com/openai/v1/models",
        ),
    ])
    def test_derives_models_url(self, api_url, expected):
        with patch("llm_cmd._API_URL", api_url):
            assert _models_url() == expected


# ── _load_models ──────────────────────────────────────────────────────────────

class TestLoadModels:
    def test_missing_cache_returns_empty(self, tmp_path):
        with patch("llm_cmd._MODELS_CACHE", tmp_path / "models.json"):
            assert _load_models() == []

    def test_valid_cache_sorted(self, tmp_path):
        cache = tmp_path / "models.json"
        cache.write_text(json.dumps({"data": [
            {"id": "openai/gpt-4o"},
            {"id": "anthropic/claude-3-5-haiku"},
        ]}))
        with patch("llm_cmd._MODELS_CACHE", cache):
            assert _load_models() == ["anthropic/claude-3-5-haiku", "openai/gpt-4o"]

    def test_corrupted_cache_returns_empty(self, tmp_path):
        cache = tmp_path / "models.json"
        cache.write_text("not json{{")
        with patch("llm_cmd._MODELS_CACHE", cache):
            assert _load_models() == []

    def test_stale_cache_still_returned(self, tmp_path):
        cache = tmp_path / "models.json"
        cache.write_text(json.dumps({"data": [{"id": "old/model"}]}))
        old = time.time() - 999_999
        os.utime(cache, (old, old))
        with patch("llm_cmd._MODELS_CACHE", cache):
            assert _load_models() == ["old/model"]


# ── _maybe_update_models_bg ───────────────────────────────────────────────────

class TestMaybeUpdateModelsBg:
    def test_skips_when_bg_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("_LLM_CMD_BG_UPDATE", "1")
        with patch("subprocess.Popen") as popen:
            with patch("llm_cmd._MODELS_CACHE", tmp_path / "missing.json"):
                _maybe_update_models_bg()
        popen.assert_not_called()

    def test_skips_when_cache_fresh(self, monkeypatch, tmp_path):
        monkeypatch.delenv("_LLM_CMD_BG_UPDATE", raising=False)
        cache = tmp_path / "models.json"
        cache.write_text("{}")
        os.utime(cache, (time.time(), time.time()))
        with patch("subprocess.Popen") as popen:
            with patch("llm_cmd._MODELS_CACHE", cache):
                _maybe_update_models_bg()
        popen.assert_not_called()

    def test_spawns_when_cache_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("_LLM_CMD_BG_UPDATE", raising=False)
        with patch("subprocess.Popen") as popen:
            with patch("llm_cmd._MODELS_CACHE", tmp_path / "missing.json"):
                _maybe_update_models_bg()
        popen.assert_called_once()

    def test_spawns_when_cache_stale(self, monkeypatch, tmp_path):
        monkeypatch.delenv("_LLM_CMD_BG_UPDATE", raising=False)
        cache = tmp_path / "models.json"
        cache.write_text("{}")
        old = time.time() - (llm_cmd._CACHE_TTL + 1)
        os.utime(cache, (old, old))
        with patch("subprocess.Popen") as popen:
            with patch("llm_cmd._MODELS_CACHE", cache):
                _maybe_update_models_bg()
        popen.assert_called_once()

    def test_spawned_process_is_detached(self, monkeypatch, tmp_path):
        monkeypatch.delenv("_LLM_CMD_BG_UPDATE", raising=False)
        with patch("subprocess.Popen") as popen:
            with patch("llm_cmd._MODELS_CACHE", tmp_path / "missing.json"):
                _maybe_update_models_bg()
        _, kwargs = popen.call_args
        assert kwargs.get("start_new_session") is True
        assert kwargs.get("stdout") == subprocess.DEVNULL
        assert kwargs.get("stdin") == subprocess.DEVNULL
        assert "_LLM_CMD_BG_UPDATE" in kwargs.get("env", {})


# ── HTTP layer ────────────────────────────────────────────────────────────────

class TestMakeRequest:
    def test_no_api_key_exits(self):
        with patch("llm_cmd._API_KEY", ""):
            with pytest.raises(SystemExit) as exc:
                llm_cmd._make_request("p", "m", None, False)
        assert exc.value.code == 1

    def test_connection_error_exits(self):
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value.request.side_effect = OSError("connection refused")
            with patch("llm_cmd._API_KEY", "key"):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd._make_request("p", "m", None, False)
        assert exc.value.code == 1

    def test_http_error_exits(self, mock_http):
        mock_http(MockHTTPResponse(401, b'{"error":"unauthorized"}'))
        with pytest.raises(SystemExit) as exc:
            llm_cmd._make_request("p", "m", None, False)
        assert exc.value.code == 1

    def test_system_message_included(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        llm_cmd._make_request("prompt", "model", "be terse", False)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        msgs = json.loads(body_arg)["messages"]
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "prompt"}

    def test_no_system_message_when_none(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        llm_cmd._make_request("prompt", "model", None, False)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        msgs = json.loads(body_arg)["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


class TestCallLlmStreaming:
    def test_prints_content(self, mock_http, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        llm_cmd.call_llm_streaming("hi", "m", None)
        assert "Hello world" in capsys.readouterr().out

    def test_skips_non_data_lines(self, mock_http, capsys):
        lines = [
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        llm_cmd.call_llm_streaming("hi", "m", None)
        assert "ok" in capsys.readouterr().out

    def test_stops_at_done(self, mock_http, capsys):
        lines = [
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}\n',
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        llm_cmd.call_llm_streaming("hi", "m", None)
        assert "SHOULD NOT APPEAR" not in capsys.readouterr().out


class TestCallLlmCapture:
    def test_returns_stripped_content(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "  result  "}}]}).encode()
        mock_http(MockHTTPResponse(200, body))
        text, stats = llm_cmd.call_llm_capture("p", "m", "sys")
        assert text == "result"
        assert stats is None  # no usage field in response

    def test_returns_usage_stats(self, mock_http):
        body = json.dumps({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0003},
        }).encode()
        mock_http(MockHTTPResponse(200, body))
        text, stats = llm_cmd.call_llm_capture("p", "m", "sys")
        assert text == "hi"
        assert stats is not None
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 20
        assert stats.cost_usd == pytest.approx(0.0003)


class TestCallLlmStreamingUsage:
    def test_captures_usage_chunk(self, mock_http, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":8,"cost":0.001}}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        stats = llm_cmd.call_llm_streaming("hi", "m", None, collect_usage=True)
        assert stats is not None
        assert stats.prompt_tokens == 5
        assert stats.completion_tokens == 8
        assert stats.cost_usd == pytest.approx(0.001)
        assert "Hello" in capsys.readouterr().out

    def test_no_usage_when_not_requested(self, mock_http, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":8}}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        stats = llm_cmd.call_llm_streaming("hi", "m", None, collect_usage=False)
        assert stats is None


# ── confirm_and_run ───────────────────────────────────────────────────────────

class TestConfirmAndRun:
    def test_y_runs_command(self):
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", return_value="y"):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd.confirm_and_run("ls -la", "list files")
        assert exc.value.code == 0
        run.assert_called_once_with("ls -la", shell=True)

    def test_empty_enter_runs_command(self):
        # Y is the default — pressing Enter should run
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", return_value=""):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd.confirm_and_run("ls", "list")
        assert exc.value.code == 0
        run.assert_called_once()

    def test_n_aborts(self):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit) as exc:
                llm_cmd.confirm_and_run("ls", "list")
        assert exc.value.code == 0

    def test_ctrl_c_aborts(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc:
                llm_cmd.confirm_and_run("ls", "list")
        assert exc.value.code == 0

    def test_fences_stripped_before_run(self):
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", return_value="y"):
                with pytest.raises(SystemExit):
                    llm_cmd.confirm_and_run("```bash\nls -la\n```", "list")
        run.assert_called_once_with("ls -la", shell=True)

    def test_e_opens_editor_then_reruns(self):
        edited_cmd = "ls -lah"
        responses = iter(["e", "y"])
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", side_effect=responses):
                with patch("llm_cmd._edit_in_editor", return_value=edited_cmd):
                    with pytest.raises(SystemExit):
                        llm_cmd.confirm_and_run("ls -la", "list files")
        run.assert_called_once_with(edited_cmd, shell=True)


# ── _edit_in_editor ───────────────────────────────────────────────────────────

class TestEditInEditor:
    def test_strips_comment_lines(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EDITOR", "true")  # 'true' command exits immediately
        # Write a temp file manually to simulate editor that doesn't change comments
        with patch("tempfile.NamedTemporaryFile") as mock_ntf:
            tmpfile = tmp_path / "cmd.sh"
            tmpfile.write_text("# Prompt: test\n# ────\n\nls -la\n")
            mock_ntf.return_value.__enter__ = lambda s: s
            mock_ntf.return_value.__exit__ = lambda *a: False
            mock_ntf.return_value.name = str(tmpfile)
            with patch("os.system"):
                with patch("os.unlink"):
                    result = llm_cmd._edit_in_editor("ls -la", "test")
        assert "#" not in result
        assert "ls -la" in result


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_missing_returns_empty(self, tmp_path):
        with patch("llm_cmd._CONFIG_FILE", tmp_path / "config.json"):
            assert llm_cmd._load_config() == {}

    def test_round_trip(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        with patch("llm_cmd._CONFIG_FILE", cfg_file), \
             patch("llm_cmd._CONFIG_DIR", tmp_path):
            llm_cmd._save_config({"default_model": "openai/gpt-4o"})
            result = llm_cmd._load_config()
        assert result == {"default_model": "openai/gpt-4o"}

    def test_load_corrupted_returns_empty(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("not json{{")
        with patch("llm_cmd._CONFIG_FILE", cfg_file):
            assert llm_cmd._load_config() == {}

    def test_resolve_env_takes_priority(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_CMD_MODEL", "env/model")
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"default_model": "config/model"}))
        with patch("llm_cmd._CONFIG_FILE", cfg_file):
            assert llm_cmd._resolve_default_model() == "env/model"

    def test_resolve_config_over_hardcoded(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_CMD_MODEL", raising=False)
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"default_model": "config/model"}))
        with patch("llm_cmd._CONFIG_FILE", cfg_file):
            assert llm_cmd._resolve_default_model() == "config/model"

    def test_resolve_hardcoded_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_CMD_MODEL", raising=False)
        with patch("llm_cmd._CONFIG_FILE", tmp_path / "missing.json"):
            assert llm_cmd._resolve_default_model() == "openai/gpt-4o-mini"


# ── History / SQLite ──────────────────────────────────────────────────────────

class TestHistory:
    def test_record_and_summary(self, tmp_path):
        stats = llm_cmd._UsageStats(
            model="openai/gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.0002,
        )
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_usage(stats, "chat")
            s = llm_cmd._cost_summary(1)
        assert s["requests"] == 1
        assert s["prompt_tokens"] == 10
        assert s["completion_tokens"] == 20
        assert s["cost_usd"] == pytest.approx(0.0002)

    def test_summary_empty_db(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "missing.db"):
            assert llm_cmd._cost_summary(7) == {}

    def test_record_never_crashes(self, tmp_path):
        stats = llm_cmd._UsageStats(model="m", prompt_tokens=1, completion_tokens=1)
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path), \
             patch("sqlite3.connect", side_effect=Exception("db error")):
            llm_cmd._record_usage(stats, "chat")  # must not raise
