"""
session_store.py — SQLite-backed persistent logging layer
==========================================================
Enables stop-and-resume semantics: evaluation jobs can be interrupted at
any point and continued exactly where they left off.

Schema overview
---------------
  sessions    – one row per red-teaming session (may span many turns)
  turns       – one row per evaluated (prompt, response) pair
  behaviors   – optional catalog of HarmBench behavior definitions

All writes use WAL mode for crash safety on the T4 host.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Generator, Optional

from judge_ensemble import EnsembleResult, JudgeVerdict


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    attack_method   TEXT,
    target_model    TEXT,
    behavior_id     TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    updated_at      REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    status          TEXT NOT NULL DEFAULT 'IN_PROGRESS',  -- IN_PROGRESS | BREACHED | EXHAUSTED
    total_turns     INTEGER NOT NULL DEFAULT 0,
    breach_turn     INTEGER,          -- TTB: first turn index at which SUCCESS was returned
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index      INTEGER NOT NULL,
    malicious_prompt TEXT NOT NULL,
    model_response  TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    stage_reached   INTEGER NOT NULL,
    refusal_detected INTEGER NOT NULL DEFAULT 0,
    llama_guard_label TEXT,
    harm_category   TEXT,
    harmbench_answer TEXT,
    latency_ms      REAL,
    recorded_at     REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    UNIQUE (session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS behaviors (
    behavior_id     TEXT PRIMARY KEY,
    category        TEXT,
    subcategory     TEXT,
    behavior_text   TEXT NOT NULL,
    attack_method   TEXT,
    source          TEXT   -- e.g. "HarmBench-standard", "AdvBench"
);

CREATE INDEX IF NOT EXISTS idx_turns_session  ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_turns_verdict  ON turns (verdict);
CREATE INDEX IF NOT EXISTS idx_sessions_model ON sessions (target_model);
"""


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """
    Thread-safe SQLite wrapper.  One instance per evaluation process.

    Usage
    -----
        store = SessionStore("results/eval_run.db")
        store.upsert_session(session_id="s1", attack_method="GCG", ...)
        store.record_turn(result, malicious_prompt, model_response)
        store.mark_breached("s1", breach_turn=3)
    """

    def __init__(self, db_path: str | Path = "harmbench_results.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        with self._conn:
            self._conn.executescript(_DDL)

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection; commit on success, rollback on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id:    str,
        attack_method: str = "",
        target_model:  str = "",
        behavior_id:   str = "",
        notes:         str = "",
    ) -> None:
        sql = """
        INSERT INTO sessions (session_id, attack_method, target_model, behavior_id, notes)
        VALUES (:session_id, :attack_method, :target_model, :behavior_id, :notes)
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at = unixepoch('now', 'subsec')
        """
        with self._tx() as con:
            con.execute(sql, {
                "session_id":    session_id,
                "attack_method": attack_method,
                "target_model":  target_model,
                "behavior_id":   behavior_id,
                "notes":         notes,
            })

    def mark_breached(self, session_id: str, breach_turn: int) -> None:
        sql = """
        UPDATE sessions
        SET status = 'BREACHED', breach_turn = :bt,
            updated_at = unixepoch('now', 'subsec')
        WHERE session_id = :sid
        """
        with self._tx() as con:
            con.execute(sql, {"bt": breach_turn, "sid": session_id})

    def mark_exhausted(self, session_id: str, total_turns: int) -> None:
        sql = """
        UPDATE sessions
        SET status = 'EXHAUSTED', total_turns = :tt,
            updated_at = unixepoch('now', 'subsec')
        WHERE session_id = :sid
        """
        with self._tx() as con:
            con.execute(sql, {"tt": total_turns, "sid": session_id})

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def already_evaluated(self, session_id: str, turn_index: int) -> bool:
        """Check whether a (session, turn) pair already has a result (resume guard)."""
        row = self._conn.execute(
            "SELECT 1 FROM turns WHERE session_id = ? AND turn_index = ?",
            (session_id, turn_index),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        result:           EnsembleResult,
        malicious_prompt: str,
        model_response:   str,
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO turns
            (session_id, turn_index, malicious_prompt, model_response,
             verdict, stage_reached, refusal_detected, llama_guard_label,
             harm_category, harmbench_answer, latency_ms)
        VALUES
            (:session_id, :turn_index, :malicious_prompt, :model_response,
             :verdict, :stage_reached, :refusal_detected, :llama_guard_label,
             :harm_category, :harmbench_answer, :latency_ms)
        """
        with self._tx() as con:
            con.execute(sql, {
                "session_id":        result.session_id or "",
                "turn_index":        result.turn_index,
                "malicious_prompt":  malicious_prompt,
                "model_response":    model_response,
                "verdict":           result.verdict.value,
                "stage_reached":     result.stage_reached,
                "refusal_detected":  int(result.refusal_detected),
                "llama_guard_label": result.llama_guard_label,
                "harm_category":     result.harm_category,
                "harmbench_answer":  result.harmbench_answer,
                "latency_ms":        result.latency_ms,
            })
            # Keep sessions.total_turns in sync
            if result.session_id:
                con.execute(
                    "UPDATE sessions SET total_turns = total_turns + 1, "
                    "updated_at = unixepoch('now', 'subsec') "
                    "WHERE session_id = ?",
                    (result.session_id,),
                )

    # ------------------------------------------------------------------
    # Behavior catalog
    # ------------------------------------------------------------------

    def insert_behavior(
        self,
        behavior_id:   str,
        behavior_text: str,
        category:      str = "",
        subcategory:   str = "",
        attack_method: str = "",
        source:        str = "HarmBench-standard",
    ) -> None:
        sql = """
        INSERT OR IGNORE INTO behaviors
            (behavior_id, category, subcategory, behavior_text, attack_method, source)
        VALUES (:behavior_id, :category, :subcategory, :behavior_text, :attack_method, :source)
        """
        with self._tx() as con:
            con.execute(sql, {
                "behavior_id":   behavior_id,
                "category":      category,
                "subcategory":   subcategory,
                "behavior_text": behavior_text,
                "attack_method": attack_method,
                "source":        source,
            })

    # ------------------------------------------------------------------
    # Analytics queries (used by MetricsEngine)
    # ------------------------------------------------------------------

    def all_turns(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Return all turn rows, optionally filtered by:
          target_model, attack_method, behavior_id, verdict.
        """
        base = """
        SELECT t.*, s.attack_method, s.target_model, s.behavior_id,
               b.category, b.subcategory
        FROM turns t
        JOIN sessions s ON t.session_id = s.session_id
        LEFT JOIN behaviors b ON s.behavior_id = b.behavior_id
        """
        clauses: list[str] = []
        params:  list[Any] = []

        if filters:
            for col in ("target_model", "attack_method", "behavior_id", "verdict"):
                if col in filters:
                    clauses.append(f"s.{col} = ?" if col != "verdict" else f"t.{col} = ?")
                    params.append(filters[col])

        if clauses:
            base += " WHERE " + " AND ".join(clauses)

        rows = self._conn.execute(base, params).fetchall()
        return [dict(r) for r in rows]

    def breach_turns(self) -> list[dict[str, Any]]:
        """Return all sessions that were breached, with TTB."""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE status = 'BREACHED'"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
