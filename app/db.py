"""SQLite persistence for raw captures and inferred chain versions."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    bsdl TEXT NOT NULL,
    model_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    kind TEXT NOT NULL,
    instruction TEXT,
    tdi TEXT NOT NULL,
    tdo TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    path = os.environ.get("JTAG_RECON_DB", "jtag_recon.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def add_device(name: str, bsdl: str, model: dict) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO devices(name, bsdl, model_json, created_at) VALUES (?,?,?,?)",
            (name, bsdl, json.dumps(model), _now()),
        )
        return cur.lastrowid


def get_device(device_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
    return dict(row) if row else None


def list_devices() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, model_json, created_at FROM devices ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["model"] = json.loads(d.pop("model_json"))
        out.append(d)
    return out


def add_capture(session: str, kind: str, instruction: str | None,
                tdi: str, tdo: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO captures(session, kind, instruction, tdi, tdo, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (session, kind, instruction, tdi, tdo, _now()),
        )
        return cur.lastrowid


def list_captures(session: str | None = None) -> list[dict]:
    with _conn() as c:
        if session:
            rows = c.execute(
                "SELECT * FROM captures WHERE session=? ORDER BY id", (session,)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM captures ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def add_version(session: str, request: dict, result: dict, note: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO versions(session, request_json, result_json, note, created_at)"
            " VALUES (?,?,?,?,?)",
            (session, json.dumps(request), json.dumps(result), note, _now()),
        )
        return cur.lastrowid


def get_version(version_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["request"] = json.loads(d.pop("request_json"))
    d["result"] = json.loads(d.pop("result_json"))
    return d


def list_versions(session: str | None = None) -> list[dict]:
    with _conn() as c:
        if session:
            rows = c.execute(
                "SELECT id, session, note, created_at FROM versions"
                " WHERE session=? ORDER BY id", (session,)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, session, note, created_at FROM versions ORDER BY id"
            ).fetchall()
    return [dict(r) for r in rows]
