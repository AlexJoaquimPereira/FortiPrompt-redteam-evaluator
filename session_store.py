"""
session_store.py — MongoDB Persistent Logging Layer
=====================================================
Replaces the original SQLite backend with a remote MongoDB server.

Collections
-----------
  sessions    — one document per red-teaming session
  turns       — one document per evaluated (prompt, response) pair
  behaviors   — optional catalog of HarmBench behavior definitions

Connection
----------
Configure via environment variable or constructor argument:

    export HARMBENCH_MONGO_URI="mongodb://user:pass@host:27017/harmbench?authSource=admin"

Or pass ``mongo_uri`` directly to ``SessionStore()``.
Falls back to ``mongodb://localhost:27017`` for local development.

Indexes
-------
  sessions : session_id (unique)
  turns    : (session_id, turn_index) (unique), verdict
  behaviors: behavior_id (unique)

Stop-and-Resume
---------------
``already_evaluated(session_id, turn_index)`` queries the unique compound
index so interrupted runs skip already-logged turns.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from judge_ensemble import EnsembleResult, JudgeVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default connection parameters
# ---------------------------------------------------------------------------

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DB  = "harmbench"


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """
    MongoDB-backed persistent store for HarmBench evaluation sessions.

    Parameters
    ----------
    mongo_uri : Full MongoDB connection URI.  If omitted, reads
                ``HARMBENCH_MONGO_URI`` env var, then falls back to
                ``mongodb://localhost:27017``.
    db_name   : Database name (default: ``"harmbench"``).
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name:   str           = _DEFAULT_DB,
    ):
        uri = mongo_uri or os.getenv("HARMBENCH_MONGO_URI", _DEFAULT_URI)
        logger.info("[MongoDB] Connecting to %s / %s …", uri.split("@")[-1], db_name)

        self._client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        # Trigger connection check immediately so misconfigured URIs surface early
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
        # sessions
        self._sessions.create_index(
            [("session_id", ASCENDING)], unique=True, background=True
        )
        self._sessions.create_index(
            [("target_model", ASCENDING)], background=True
        )
        # turns
        self._turns.create_index(
            [("session_id", ASCENDING), ("turn_index", ASCENDING)],
            unique=True, background=True,
        )
        self._turns.create_index([("verdict", ASCENDING)], background=True)
        # behaviors
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
        """Insert a new session document or touch ``updated_at`` if it exists."""
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
                "status":       "BREACHED",
                "breach_turn":  breach_turn,
                "updated_at":   time.time(),
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
        return doc  # type: ignore[return-value]

    def already_evaluated(self, session_id: str, turn_index: int) -> bool:
        """True if (session_id, turn_index) already has a turn document."""
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
    ) -> None:
        """Insert a turn document.  Uses upsert to be idempotent on retries."""
        now = time.time()
        doc: dict[str, Any] = {
            "session_id":         result.session_id or "",
            "turn_index":         result.turn_index,
            "malicious_prompt":   malicious_prompt,
            "model_response":     model_response,
            "verdict":            result.verdict.value,
            "stage_reached":      result.stage_reached,
            "refusal_detected":   result.refusal_detected,
            "tool_calls_found":   result.tool_calls_found,
            "raw_judge_output":   result.raw_judge_output,
            "latency_ms":         result.latency_ms,
            "recorded_at":        now,
            # Flattened TriLabels (None-safe)
            "intent_harm":        result.labels.intent_harm    if result.labels else None,
            "response_harm":      result.labels.response_harm  if result.labels else None,
            "refusal_signal":     result.labels.refusal_signal if result.labels else None,
            "label_parse_error":  result.labels.parse_error    if result.labels else None,
        }
        try:
            self._turns.update_one(
                {"session_id": doc["session_id"], "turn_index": doc["turn_index"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
        except DuplicateKeyError:
            pass  # Already present — resume guard is working correctly

        # Increment session turn counter
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
        Return enriched turn documents joined with session and behavior data.

        Supported filter keys: target_model, attack_method, behavior_id, verdict.
        """
        # Aggregate pipeline: turns → join sessions → join behaviors
        pipeline: list[dict] = [
            {
                "$lookup": {
                    "from":         "sessions",
                    "localField":   "session_id",
                    "foreignField": "session_id",
                    "as":           "_session",
                }
            },
            {"$unwind": {"path": "$_session", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from":         "behaviors",
                    "localField":   "_session.behavior_id",
                    "foreignField": "behavior_id",
                    "as":           "_behavior",
                }
            },
            {"$unwind": {"path": "$_behavior", "preserveNullAndEmptyArrays": True}},
            {
                "$addFields": {
                    "attack_method": "$_session.attack_method",
                    "target_model":  "$_session.target_model",
                    "behavior_id":   "$_session.behavior_id",
                    "category":      "$_behavior.category",
                    "subcategory":   "$_behavior.subcategory",
                }
            },
            {"$project": {"_id": 0, "_session": 0, "_behavior": 0}},
        ]

        # Apply filters as a $match stage at the start
        if filters:
            match: dict[str, Any] = {}
            if "verdict" in filters:
                match["verdict"] = filters["verdict"]
            pipeline.insert(0, {"$match": match})

            # Session-level filters applied after lookup
            session_match: dict[str, Any] = {}
            for key in ("target_model", "attack_method", "behavior_id"):
                if key in filters:
                    session_match[f"_session.{key}"] = filters[key]
            if session_match:
                pipeline.insert(
                    next(i for i, s in enumerate(pipeline) if "$unwind" in s) + 1,
                    {"$match": session_match},
                )

        return list(self._turns.aggregate(pipeline))

    def breach_turns(self) -> list[dict[str, Any]]:
        """Return all session documents with status BREACHED."""
        return list(
            self._sessions.find({"status": "BREACHED"}, {"_id": 0})
        )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()
        logger.info("[MongoDB] Connection closed.")
