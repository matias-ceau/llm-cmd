import json
import os
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm_cmd
from llm_cmd import (
    _UsageStats,
    _build_user_content,
    _execute_prompt,
    _is_image_url,
    _load_models,
    _maybe_update_models_bg,
    _models_url,
    _strip_fences,
    build_parser,
    get_content,
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


# ── build_parser / get_content ────────────────────────────────────────────────

class TestParser:
    def test_words_joined(self):
        args = build_parser().parse_args(["what", "is", "X"])
        content, mods = get_content(args)
        assert content == "what is X"
        assert mods == set()

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
        args = build_parser().parse_args(["-S", "be terse", "hi"])
        assert args.system == "be terse"

    def test_update_models_flag(self):
        assert build_parser().parse_args(["--update-models"]).update_models is True

    def test_list_models_flag(self):
        assert build_parser().parse_args(["--list-models"]).list_models is True

    def test_session_flag(self):
        args = build_parser().parse_args(["-s", "myconv", "hi"])
        assert args.session == "myconv"

    def test_session_auto(self):
        args = build_parser().parse_args(["-s", "auto", "hi"])
        assert args.session == "auto"

    def test_follow_up_flag(self):
        assert build_parser().parse_args(["-f", "hi"]).follow_up is True

    def test_input_flag(self):
        args = build_parser().parse_args(["-i", "photo.jpg", "describe"])
        assert args.input == ["photo.jpg"]

    def test_input_flag_repeatable(self):
        args = build_parser().parse_args(["-i", "a.jpg", "-i", "b.png", "describe"])
        assert args.input == ["a.jpg", "b.png"]

    def test_quiet_flag(self):
        assert build_parser().parse_args(["-q", "hi"]).quiet is True


class TestGetContent:
    def test_stdin_fallback(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = "  piped prompt\n"
        with patch("sys.stdin", stdin):
            content, mods = get_content(args)
        assert content == "piped prompt"
        assert mods == set()

    def test_tty_no_words_exits(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = True
        with patch("sys.stdin", stdin):
            with pytest.raises(SystemExit) as exc:
                get_content(args)
        assert exc.value.code == 1

    def test_empty_stdin_exits(self):
        args = build_parser().parse_args([])
        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = "   "
        with patch("sys.stdin", stdin):
            with pytest.raises(SystemExit) as exc:
                get_content(args)
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
    def _msgs(self, prompt="p", system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def test_no_api_key_exits(self):
        with patch("llm_cmd._API_KEY", ""):
            with pytest.raises(SystemExit) as exc:
                llm_cmd._make_request(self._msgs(), "m", False)
        assert exc.value.code == 1

    def test_connection_error_exits(self):
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value.request.side_effect = OSError("connection refused")
            with patch("llm_cmd._API_KEY", "key"):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd._make_request(self._msgs(), "m", False)
        assert exc.value.code == 1

    def test_http_error_exits(self, mock_http):
        mock_http(MockHTTPResponse(401, b'{"error":"unauthorized"}'))
        with pytest.raises(SystemExit) as exc:
            llm_cmd._make_request(self._msgs(), "m", False)
        assert exc.value.code == 1

    def test_system_message_included(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        msgs = self._msgs("prompt", "be terse")
        llm_cmd._make_request(msgs, "model", False)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        sent_msgs = json.loads(body_arg)["messages"]
        assert sent_msgs[0] == {"role": "system", "content": "be terse"}
        assert sent_msgs[1] == {"role": "user", "content": "prompt"}

    def test_no_system_message_when_none(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        msgs = self._msgs("prompt")
        llm_cmd._make_request(msgs, "model", False)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        sent_msgs = json.loads(body_arg)["messages"]
        assert len(sent_msgs) == 1
        assert sent_msgs[0]["role"] == "user"

    def test_stream_options_added_when_streaming(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        llm_cmd._make_request(self._msgs(), "m", stream=True, include_usage=True)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        assert json.loads(body_arg).get("stream_options") == {"include_usage": True}

    def test_no_stream_options_without_flag(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        http_cls = mock_http(MockHTTPResponse(200, body))
        llm_cmd._make_request(self._msgs(), "m", stream=True, include_usage=False)
        _, _, body_arg, _ = http_cls.return_value.request.call_args[0]
        assert "stream_options" not in json.loads(body_arg)


class TestCallLlmStreaming:
    def _msgs(self):
        return [{"role": "user", "content": "hi"}]

    def test_prints_content(self, mock_http, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        text, _ = llm_cmd.call_llm_streaming(self._msgs(), "m")
        assert "Hello world" in capsys.readouterr().out
        assert text == "Hello world"

    def test_skips_non_data_lines(self, mock_http, capsys):
        lines = [
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        text, _ = llm_cmd.call_llm_streaming(self._msgs(), "m")
        assert "ok" in capsys.readouterr().out

    def test_stops_at_done(self, mock_http, capsys):
        lines = [
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}\n',
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        llm_cmd.call_llm_streaming(self._msgs(), "m")
        assert "SHOULD NOT APPEAR" not in capsys.readouterr().out

    def test_captures_usage_chunk(self, mock_http, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":8,"cost":0.001}}\n',
            b"data: [DONE]\n",
        ]
        mock_http(MockHTTPResponse(200, b"", lines))
        text, stats = llm_cmd.call_llm_streaming(self._msgs(), "m", collect_usage=True)
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
        _, stats = llm_cmd.call_llm_streaming(self._msgs(), "m", collect_usage=False)
        assert stats is None


class TestCallLlmCapture:
    def _msgs(self):
        return [{"role": "user", "content": "p"}]

    def test_returns_stripped_content(self, mock_http):
        body = json.dumps({"choices": [{"message": {"content": "  result  "}}]}).encode()
        mock_http(MockHTTPResponse(200, body))
        text, stats = llm_cmd.call_llm_capture(self._msgs(), "m")
        assert text == "result"
        assert stats is None

    def test_returns_usage_stats(self, mock_http):
        body = json.dumps({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0003},
        }).encode()
        mock_http(MockHTTPResponse(200, body))
        text, stats = llm_cmd.call_llm_capture(self._msgs(), "m")
        assert text == "hi"
        assert stats is not None
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 20
        assert stats.cost_usd == pytest.approx(0.0003)


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
        monkeypatch.setenv("EDITOR", "true")
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
        stats = _UsageStats(
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
        stats = _UsageStats(model="m", prompt_tokens=1, completion_tokens=1)
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path), \
             patch("sqlite3.connect", side_effect=Exception("db error")):
            llm_cmd._record_usage(stats, "chat")  # must not raise


# ── Sessions ──────────────────────────────────────────────────────────────────

class TestSessions:
    def test_record_and_retrieve_messages(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_message("sess1", "user",      "hello",    None,  None, "chat")
            llm_cmd._record_message("sess1", "assistant", "world",    "gpt", None, "chat")
            msgs = llm_cmd._get_session_messages("sess1")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user",      "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "world"}

    def test_last_session_id(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_message("first",  "user", "a", None, None, "chat")
            llm_cmd._record_message("second", "user", "b", None, None, "chat")
            last = llm_cmd._last_session_id()
        assert last == "second"

    def test_last_session_none_when_empty(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "missing.db"):
            assert llm_cmd._last_session_id() is None

    def test_resolve_session_none(self):
        sid, msgs = llm_cmd._resolve_session(None, False)
        assert sid is None
        assert msgs == []

    def test_resolve_session_named_new(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            sid, msgs = llm_cmd._resolve_session("myconv", False)
        assert sid == "myconv"
        assert msgs == []

    def test_resolve_session_named_existing(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_message("myconv", "user",      "hi",  None,  None, "chat")
            llm_cmd._record_message("myconv", "assistant", "hey", "gpt", None, "chat")
            sid, msgs = llm_cmd._resolve_session("myconv", False)
        assert sid == "myconv"
        assert len(msgs) == 2

    def test_resolve_session_auto_generates_name(self, tmp_path, capsys):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            sid, msgs = llm_cmd._resolve_session("auto", False)
        assert sid is not None
        assert sid.startswith("auto-")
        assert msgs == []
        assert "Session:" in capsys.readouterr().err

    def test_resolve_follow_up(self, tmp_path, capsys):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_message("prev", "user",      "q", None,  None, "chat")
            llm_cmd._record_message("prev", "assistant", "a", "gpt", None, "chat")
            sid, msgs = llm_cmd._resolve_session(None, follow_up=True)
        assert sid == "prev"
        assert len(msgs) == 2

    def test_resolve_follow_up_no_history_exits(self, tmp_path):
        with patch("llm_cmd._HISTORY_DB", tmp_path / "missing.db"):
            with pytest.raises(SystemExit):
                llm_cmd._resolve_session(None, follow_up=True)

    def test_session_and_followup_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            llm_cmd._resolve_session("myconv", follow_up=True)

    def test_multimodal_content_round_trip(self, tmp_path):
        multimodal = [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        with patch("llm_cmd._HISTORY_DB", tmp_path / "history.db"), \
             patch("llm_cmd._DATA_DIR", tmp_path):
            llm_cmd._record_message("mm", "user", multimodal, None, None, "chat")
            msgs = llm_cmd._get_session_messages("mm")
        assert msgs[0]["content"] == multimodal


# ── Multimodal ────────────────────────────────────────────────────────────────

class TestIsImageUrl:
    def test_http_jpg(self):
        assert _is_image_url("https://example.com/photo.jpg")

    def test_https_png(self):
        assert _is_image_url("https://example.com/img.png")

    def test_not_an_image_url(self):
        assert not _is_image_url("https://example.com/doc.pdf")

    def test_not_a_url(self):
        assert not _is_image_url("photo.jpg")

    def test_http_not_https(self):
        assert _is_image_url("http://example.com/a.webp")


class TestBuildUserContent:
    def test_plain_text_words(self):
        content, mods = _build_user_content(["what", "is", "this"])
        assert content == "what is this"
        assert mods == set()

    def test_image_url_in_words(self):
        content, mods = _build_user_content(["describe", "https://example.com/photo.jpg"])
        assert isinstance(content, list)
        assert mods == {"image"}
        text_part = next(p for p in content if p.get("type") == "text")
        assert text_part["text"] == "describe"
        img_part = next(p for p in content if p.get("type") == "image_url")
        assert img_part["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_local_image_file(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")  # minimal PNG header
        content, mods = _build_user_content(["describe", str(img)])
        assert isinstance(content, list)
        assert "image" in mods
        img_part = next(p for p in content if p.get("type") == "image_url")
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_explicit_file_via_extra(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # JPEG magic bytes
        content, mods = _build_user_content(["what is this"], [str(img)])
        assert isinstance(content, list)
        assert "image" in mods

    def test_nonexistent_explicit_file_warned(self, tmp_path, capsys):
        _build_user_content(["hi"], ["/nonexistent/file.jpg"])
        assert "not found" in capsys.readouterr().err

    def test_non_media_file_ignored(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        content, mods = _build_user_content(["read", str(txt)])
        # .txt is not in _MEDIA_EXTENSIONS, treated as text token
        assert isinstance(content, str)
        assert mods == set()

    def test_text_only_returns_str(self):
        content, mods = _build_user_content(["hello", "world"])
        assert isinstance(content, str)
        assert content == "hello world"

    def test_only_image_no_text(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")
        content, mods = _build_user_content([str(img)])
        assert isinstance(content, list)
        # No text part when only a file
        text_parts = [p for p in content if p.get("type") == "text"]
        assert text_parts == []


class TestEncodeFileContent:
    def test_image_format(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"PNG")
        part = llm_cmd._encode_file_content(f)
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_pdf_format(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        part = llm_cmd._encode_file_content(f)
        assert part["type"] == "file"
        assert part["file"]["filename"] == "doc.pdf"

    def test_audio_format(self, tmp_path):
        f = tmp_path / "sound.mp3"
        f.write_bytes(b"ID3")
        part = llm_cmd._encode_file_content(f)
        assert part["type"] == "input_audio"
        assert part["input_audio"]["format"] == "mp3"

    def test_video_format(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00\x00\x00\x18")
        part = llm_cmd._encode_file_content(f)
        assert part["type"] == "video_url"
        assert part["video_url"]["url"].startswith("data:video/mp4;base64,")


class TestModalitySupport:
    def _make_cache(self, tmp_path, models):
        cache = tmp_path / "models.json"
        cache.write_text(json.dumps({"data": models}))
        return cache

    def test_supported_modality_no_error(self, tmp_path):
        cache = self._make_cache(tmp_path, [
            {"id": "m", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}}
        ])
        with patch("llm_cmd._MODELS_CACHE", cache):
            llm_cmd._check_modality_support("m", {"image"})  # should not raise

    def test_unsupported_modality_exits(self, tmp_path, capsys):
        cache = self._make_cache(tmp_path, [
            {"id": "m", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}
        ])
        with patch("llm_cmd._MODELS_CACHE", cache):
            with pytest.raises(SystemExit) as exc:
                llm_cmd._check_modality_support("m", {"image"})
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "image" in err

    def test_empty_needed_no_error(self, tmp_path):
        llm_cmd._check_modality_support("any", set())  # should not raise

    def test_list_by_modality(self, tmp_path):
        cache = self._make_cache(tmp_path, [
            {"id": "img-model", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
            {"id": "text-only", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
            {"id": "audio-gen", "architecture": {"input_modalities": ["text"], "output_modalities": ["text", "audio"]}},
        ])
        with patch("llm_cmd._MODELS_CACHE", cache):
            img_models = llm_cmd._list_models_by_modality(in_mods=["image"])
            audio_out  = llm_cmd._list_models_by_modality(out_mods=["audio"])
        assert "img-model" in img_models
        assert "text-only" not in img_models
        assert "audio-gen" in audio_out
        assert "text-only" not in audio_out
