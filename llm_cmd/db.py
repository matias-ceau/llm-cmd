import json
import sqlite3
import sys
import time
from dataclasses import dataclass

from . import constants


@dataclass
class _UsageStats:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None


def _db_conn() -> sqlite3.Connection:
    constants._DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(constants._HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            ts                REAL    NOT NULL,
            model             TEXT    NOT NULL,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd          REAL,
            mode              TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON history(ts)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        TEXT    NOT NULL,
            ts                REAL    NOT NULL,
            role              TEXT    NOT NULL,
            content           TEXT    NOT NULL,
            model             TEXT,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd          REAL,
            mode              TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)")
    conn.commit()
    return conn


def _record_usage(stats: _UsageStats, mode: str) -> None:
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO history(ts,model,prompt_tokens,completion_tokens,cost_usd,mode)"
                " VALUES(?,?,?,?,?,?)",
                (time.time(), stats.model, stats.prompt_tokens,
                 stats.completion_tokens, stats.cost_usd, mode),
            )
    except Exception:
        pass


def _record_message(
    session_id: str,
    role: str,
    content: str | list,
    model: str | None,
    stats: _UsageStats | None,
    mode: str | None,
) -> None:
    content_str = json.dumps(content) if isinstance(content, list) else content
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO messages"
                "(session_id,ts,role,content,model,prompt_tokens,completion_tokens,cost_usd,mode)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id, time.time(), role, content_str, model,
                    stats.prompt_tokens if stats else 0,
                    stats.completion_tokens if stats else 0,
                    stats.cost_usd if stats else None,
                    mode,
                ),
            )
    except Exception:
        pass


def _get_session_messages(session_id: str) -> list[dict]:
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY ts",
                (session_id,),
            ).fetchall()
    except Exception:
        return []
    result = []
    for role, content in rows:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                result.append({"role": role, "content": parsed})
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        result.append({"role": role, "content": content})
    return result


def _last_session_id() -> str | None:
    if not constants._HISTORY_DB.exists():
        return None
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT session_id FROM messages ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _resolve_session(
    session_arg: str | None,
    follow_up: bool,
) -> tuple[str | None, list[dict]]:
    if session_arg and follow_up:
        print("Error: --session and --follow-up are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if follow_up:
        sid = _last_session_id()
        if not sid:
            print("No previous session found.", file=sys.stderr)
            sys.exit(1)
        print(f"\033[2mSession: {sid}\033[0m", file=sys.stderr)
        return sid, _get_session_messages(sid)

    if session_arg is None:
        return None, []

    if session_arg == "auto":
        sid = "auto-" + time.strftime("%Y%m%dT%H%M%S")
        print(f"\033[2mSession: {sid}\033[0m", file=sys.stderr)
        return sid, []

    # Named session — load history if it exists
    msgs = _get_session_messages(session_arg)
    return session_arg, msgs


def _cost_summary(days: int) -> dict:
    if not constants._HISTORY_DB.exists():
        return {}
    since = 0.0 if days == 0 else time.time() - days * 86400
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(cost_usd), SUM(prompt_tokens), SUM(completion_tokens)"
                " FROM history WHERE ts > ?",
                (since,),
            ).fetchone()
    except Exception:
        return {}
    return {
        "requests":          row[0] or 0,
        "cost_usd":          row[1] or 0.0,
        "prompt_tokens":     row[2] or 0,
        "completion_tokens": row[3] or 0,
    }
