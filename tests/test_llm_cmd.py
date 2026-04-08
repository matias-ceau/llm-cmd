import json
import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import llm_cmd
from llm_cmd import (
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
        """Stale data is served — background refresh handles updates, not _load_models."""
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

    def test_http_error_exits(self):
        resp = MockHTTPResponse(401, b'{"error":"unauthorized"}')
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(resp)
            with patch("llm_cmd._API_KEY", "key"):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd._make_request("p", "m", None, False)
        assert exc.value.code == 1

    def test_system_message_included(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        resp = MockHTTPResponse(200, body)
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(resp)
            with patch("llm_cmd._API_KEY", "key"):
                llm_cmd._make_request("prompt", "model", "be terse", False)
        _, _, body_arg, _ = cls.return_value.request.call_args[0]
        msgs = json.loads(body_arg)["messages"]
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "prompt"}

    def test_no_system_message_when_none(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        resp = MockHTTPResponse(200, body)
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(resp)
            with patch("llm_cmd._API_KEY", "key"):
                llm_cmd._make_request("prompt", "model", None, False)
        _, _, body_arg, _ = cls.return_value.request.call_args[0]
        msgs = json.loads(body_arg)["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


class TestCallLlmStreaming:
    def test_prints_content(self, capsys):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b"data: [DONE]\n",
        ]
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(MockHTTPResponse(200, b"", lines))
            with patch("llm_cmd._API_KEY", "key"):
                llm_cmd.call_llm_streaming("hi", "m", None)
        assert "Hello world" in capsys.readouterr().out

    def test_skips_non_data_lines(self, capsys):
        lines = [
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(MockHTTPResponse(200, b"", lines))
            with patch("llm_cmd._API_KEY", "key"):
                llm_cmd.call_llm_streaming("hi", "m", None)
        assert "ok" in capsys.readouterr().out

    def test_stops_at_done(self, capsys):
        lines = [
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}\n',
        ]
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(MockHTTPResponse(200, b"", lines))
            with patch("llm_cmd._API_KEY", "key"):
                llm_cmd.call_llm_streaming("hi", "m", None)
        assert "SHOULD NOT APPEAR" not in capsys.readouterr().out


class TestCallLlmCapture:
    def test_returns_stripped_content(self):
        body = json.dumps({"choices": [{"message": {"content": "  result  "}}]}).encode()
        with patch("http.client.HTTPSConnection") as cls:
            cls.return_value = _mock_conn(MockHTTPResponse(200, body))
            with patch("llm_cmd._API_KEY", "key"):
                result = llm_cmd.call_llm_capture("p", "m", "sys")
        assert result == "result"


# ── confirm_and_run ───────────────────────────────────────────────────────────

class TestConfirmAndRun:
    def test_y_runs_command(self):
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", return_value="y"):
                with pytest.raises(SystemExit) as exc:
                    llm_cmd.confirm_and_run("ls -la")
        assert exc.value.code == 0
        run.assert_called_once_with("ls -la", shell=True)

    def test_n_aborts(self):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit) as exc:
                llm_cmd.confirm_and_run("ls")
        assert exc.value.code == 0

    def test_empty_enter_aborts(self):
        with patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit) as exc:
                llm_cmd.confirm_and_run("ls")
        assert exc.value.code == 0

    def test_ctrl_c_aborts(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc:
                llm_cmd.confirm_and_run("ls")
        assert exc.value.code == 0

    def test_fences_stripped_before_run(self):
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", return_value="y"):
                with pytest.raises(SystemExit):
                    llm_cmd.confirm_and_run("```bash\nls -la\n```")
        run.assert_called_once_with("ls -la", shell=True)

    def test_e_opens_editor_then_reruns(self, tmp_path):
        edited_cmd = "ls -lah"

        def fake_editor(path):
            # Write the edited command to the tmp file
            import re
            m = re.search(r'"(.+?)"', path)
            if m:
                with open(m.group(1), "w") as f:
                    f.write(edited_cmd)

        responses = iter(["e", "y"])
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            with patch("builtins.input", side_effect=responses):
                with patch("os.system", side_effect=fake_editor):
                    with pytest.raises(SystemExit):
                        llm_cmd.confirm_and_run("ls -la")
        run.assert_called_once_with(edited_cmd, shell=True)
