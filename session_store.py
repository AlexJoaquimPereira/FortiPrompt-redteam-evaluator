"""
session_store.py — MongoDB Persistent Logging Layer
=====================================================
Collections
-----------
  sessions    — one document per red-teaming session
  turns       — one document per evaluated (prompt, response) pair
  behaviors   — HarmBench behavior catalog

Connection
----------
    export HARMBENCH_MONGO_URI="mongodb://user:pass@host:27017/harmbench"
or pass ``mongo_uri`` to the constructor.  Falls back to localhost.

Stop-and-Resume
---------------
``already_evaluated()`` queries the unique (session_id, turn_index) index
so interrupted runs skip already-logged turns without re-evaluating.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from judge_ensemble import EnsembleResult, JudgeVerdict

logger = logging.getLogger(__name__)

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DB  = "harmbench"


class SessionStore:
    """
    MongoDB-backed store for HarmBench evaluation sessions.

    Parameters
    ----------
    mongo_uri : Connection URI (reads HARMBENCH_MONGO_URI env if omitted).
    db_name   : Database name (default: "harmbench").
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name:   str           = _DEFAULT_DB,
    ):
        uri = mongo_uri or os.getenv("HARMBENCH_MONGO_URI", _DEFAULT_URI)
        # Mask credentials in the log
        log_uri = uri.split("@")[-1] if "@" in uri else uri
        logger.info("[MongoDB] Connecting to %s / %s …", log_uri, db_name)

        self._client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        self._client.admin.command("ping")
        logger.info("[MongoDB] Connected.")

        db = self._client[db_name]
        self._sessions:  Collection = db["sessions"]
        self._turns:     Collection = db["turns"]
        self._behaviors: Collection = db["behaviors"]

        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Index creation (idempotent)
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        self._sessions.create_index(
            [("session_id", ASCENDING)], unique=True, background=True
        )
        self._sessions.create_index([("target_model", ASCENDING)], background=True)
        self._sessions.create_index([("attack_method", ASCENDING)], background=True)
        self._turns.create_index(
            [("session_id", ASCENDING), ("turn_index", ASCENDING)],
            unique=True, background=True,
        )
        self._turns.create_index([("verdict", ASCENDING)],   background=True)
        self._turns.create_index([("is_benign", ASCENDING)], background=True)
        self._behaviors.create_index(
            [("behavior_id", ASCENDING)], unique=True, background=True
        )
        logger.debug("[MongoDB] Indexes ensured.")

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
        """Insert a new session document, or touch updated_at if it exists."""
        now = time.time()
        self._sessions.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id":    session_id,
                    "attack_method": attack_method,
                    "target_model":  target_model,
                    "behavior_id":   behavior_id,
                    "notes":         notes,
                    "status":        "IN_PROGRESS",
                    "total_turns":   0,
                    "breach_turn":   None,
                    "created_at":    now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )

    def mark_breached(self, session_id: str, breach_turn: int) -> None:
        self._sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "status":      "BREACHED",
                "breach_turn": breach_turn,
                "updated_at":  time.time(),
            }},
        )

    def mark_exhausted(self, session_id: str, total_turns: int) -> None:
        self._sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "status":      "EXHAUSTED",
                "total_turns": total_turns,
                "updated_at":  time.time(),
            }},
        )

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        doc = self._sessions.find_one({"session_id": session_id}, {"_id": 0})
        return dict(doc) if doc else None

    def already_evaluated(self, session_id: str, turn_index: int) -> bool:
        """True if this (session_id, turn_index) pair already has a turn document."""
        return self._turns.count_documents(
            {"session_id": session_id, "turn_index": turn_index},
            limit=1,
        ) > 0

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        result:           EnsembleResult,
        malicious_prompt: str,
        model_response:   str,
        is_benign:        bool = False,
    ) -> None:
        """
        Persist one turn result.  is_benign=True marks benign probe turns
        so compute_refusal_robustness() can calculate over-refusal rates.
        Uses upsert with $setOnInsert to be idempotent on retries.
        """
        now = time.time()
        doc: dict[str, Any] = {
            "session_id":        result.session_id or "",
            "turn_index":        result.turn_index,
            "malicious_prompt":  malicious_prompt,
            "model_response":    model_response,
            "verdict":           result.verdict.value,
            "stage_reached":     result.stage_reached,
            "refusal_detected":  result.refusal_detected,
            "tool_calls_found":  result.tool_calls_found,
            "raw_judge_output":  result.raw_judge_output,
            "latency_ms":        result.latency_ms,
            "recorded_at":       now,
            "is_benign":         is_benign,
            # TriLabels — flattened, None-safe
            "intent_harm":       result.labels.intent_harm    if result.labels else None,
            "response_harm":     result.labels.response_harm  if result.labels else None,
            "refusal_signal":    result.labels.refusal_signal if result.labels else None,
            "label_parse_error": result.labels.parse_error    if result.labels else None,
        }
        try:
            self._turns.update_one(
                {"session_id": doc["session_id"], "turn_index": doc["turn_index"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
        except DuplicateKeyError:
            pass  # Already present — resume guard is correct

        if result.session_id:
            self._sessions.update_one(
                {"session_id": result.session_id},
                {"$inc": {"total_turns": 1}, "$set": {"updated_at": now}},
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
        self._behaviors.update_one(
            {"behavior_id": behavior_id},
            {"$setOnInsert": {
                "behavior_id":   behavior_id,
                "behavior_text": behavior_text,
                "category":      category,
                "subcategory":   subcategory,
                "attack_method": attack_method,
                "source":        source,
            }},
            upsert=True,
        )

    # ------------------------------------------------------------------
    # Analytics queries (used by MetricsEngine)
    # ------------------------------------------------------------------

    def all_turns(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Return enriched turn documents joined with session + behavior data.

        Filter keys supported: target_model, attack_method, behavior_id, verdict.

        Pipeline: turns → $lookup sessions → $lookup behaviors → $addFields
        Filters applied as early $match stages so MongoDB can use indexes.
        """
        pipeline: list[dict[str, Any]] = []

        # Turn-level pre-filter (applied before any lookups)
        turn_match: dict[str, Any] = {}
        if filters and "verdict" in filters:
            turn_match["verdict"] = filters["verdict"]
        if turn_match:
            pipeline.append({"$match": turn_match})

        # Join sessions
        pipeline += [
            {"$lookup": {
                "from":         "sessions",
                "localField":   "session_id",
                "foreignField": "session_id",
                "as":           "_session",
            }},
            {"$unwind": {"path": "$_session", "preserveNullAndEmptyArrays": True}},
        ]

        # Session-level post-join filter
        session_match: dict[str, Any] = {}
        if filters:
            for key in ("target_model", "attack_method", "behavior_id"):
                if key in filters:
                    session_match[f"_session.{key}"] = filters[key]
        if session_match:
            pipeline.append({"$match": session_match})

        # Join behaviors
        pipeline += [
            {"$lookup": {
                "from":         "behaviors",
                "localField":   "_session.behavior_id",
                "foreignField": "behavior_id",
                "as":           "_behavior",
            }},
            {"$unwind": {"path": "$_behavior", "preserveNullAndEmptyArrays": True}},
            {"$addFields": {
                "attack_method": "$_session.attack_method",
                "target_model":  "$_session.target_model",
                "behavior_id":   "$_session.behavior_id",
                "category":      "$_behavior.category",
                "subcategory":   "$_behavior.subcategory",
            }},
            {"$project": {"_id": 0, "_session": 0, "_behavior": 0}},
        ]

        return list(self._turns.aggregate(pipeline))

    def breach_turns(self) -> list[dict[str, Any]]:
        """Return all session documents with status BREACHED."""
        return list(self._sessions.find({"status": "BREACHED"}, {"_id": 0}))

    def get_all_sessions(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return all session documents, optionally filtered."""
        query = {k: v for k, v in (filters or {}).items()}
        return list(self._sessions.find(query, {"_id": 0}))

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()
        logger.info("[MongoDB] Connection closed.")