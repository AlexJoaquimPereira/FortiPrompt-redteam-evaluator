"""
api_server.py — FortiPrompt RedTeam Evaluator API
==================================================
FastAPI application exposing the evaluation pipeline over HTTP and WebSocket.

REST endpoints
--------------
  POST /evaluate          Synchronous batch evaluation (returns full results)
  POST /evaluate/session  Single session evaluation, returns SessionSummary JSON
  GET  /report            Metrics report from MongoDB (ASR, TTB, RR, Heatmap)
  GET  /health            Liveness probe — returns judge and DB status
  GET  /sessions          List stored sessions with optional filters
  GET  /sessions/{id}     Get one session by ID

WebSocket endpoint
------------------
  WS  /ws/evaluate        Streaming evaluation — sends one JSON message per turn,
                          then a final summary message

Input contract
--------------
All evaluation endpoints accept plain string lists — NO OpenAI schema:

    {
      "prompts":   ["user turn 1", "user turn 2", ...],
      "responses": ["model reply 1", "model reply 2", ...],
      "attack_method": "GCG",
      "target_model":  "llama-3-8b",
      "behavior_id":   "explosives_001",
      "benign_turns":  [0],
      "session_id":    "optional-uuid"
    }

Throughput configuration
-------------------------
  JUDGE_MAX_BATCH    (env int, default 16)  — pairs per GPU forward pass
  JUDGE_LAZY_LOAD    (env bool, default 0)  — keep model resident (0) or evict (1)
  MONGO_URI          (env str)              — MongoDB connection string
  MONGO_PERSIST      (env bool, default 1)  — write results to MongoDB
  MAX_CONCURRENT_WS  (env int, default 8)   — max simultaneous WebSocket sessions

Run
---
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

Note: use ``--workers 1`` — the GPU judge is a singleton shared across
requests. Multiple worker processes would each load a separate model copy
and exhaust VRAM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from evaluator import MultiTurnEvaluator, SessionSummary, TurnResult
from judge_ensemble import EnsembleJudge

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

_MONGO_URI         = os.getenv("MONGO_URI") or os.getenv("HARMBENCH_MONGO_URI")
_MONGO_PERSIST     = os.getenv("MONGO_PERSIST", "1").strip() not in ("0", "false", "False")
_JUDGE_LAZY_LOAD   = os.getenv("JUDGE_LAZY_LOAD", "0").strip() in ("1", "true", "True")
_JUDGE_MAX_BATCH   = int(os.getenv("JUDGE_MAX_BATCH", "16"))
_MAX_CONCURRENT_WS = int(os.getenv("MAX_CONCURRENT_WS", "8"))
_WS_SEMAPHORE      = asyncio.Semaphore(_MAX_CONCURRENT_WS)

# ---------------------------------------------------------------------------
# Global singletons (initialised in lifespan)
# ---------------------------------------------------------------------------

_judge:     Optional[EnsembleJudge]     = None
_evaluator: Optional[MultiTurnEvaluator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the WildGuard model and open the DB connection at startup."""
    global _judge, _evaluator

    logger.info("=== FortiPrompt API — Starting up ===")
    _judge = EnsembleJudge(lazy_load=_JUDGE_LAZY_LOAD, max_batch=_JUDGE_MAX_BATCH)
    _evaluator = MultiTurnEvaluator(
        mongo_uri=_MONGO_URI,
        judge=_judge,
        persist=_MONGO_PERSIST,
    )
    logger.info("=== Startup complete. Ready to serve. ===")

    yield   # ← server is running

    logger.info("=== Shutting down ===")
    if _evaluator:
        _evaluator.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FortiPrompt RedTeam Evaluator API",
    description=(
        "Evaluates LLM safety using the WildGuard-7B oracle. "
        "Accepts plain (prompts, responses) lists — no OpenAI schema required."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Input for a multi-turn evaluation session."""
    prompts:       list[str] = Field(..., min_length=1, description="Ordered attacker / user prompts.")
    responses:     list[str] = Field(..., min_length=1, description="Ordered target model responses.")
    attack_method: str       = Field("",  description="Attack name, e.g. 'GCG', 'PAIR', 'DirectRequest'.")
    target_model:  str       = Field("",  description="Name/id of the model under evaluation.")
    behavior_id:   str       = Field("",  description="HarmBench behavior identifier.")
    benign_turns:  list[int] = Field(default_factory=lambda: [0], description="0-based indices of benign probe turns.")
    session_id:    Optional[str] = Field(None, description="Existing session UUID to resume, or null to auto-generate.")
    persist:       Optional[bool] = Field(None, description="Override global MONGO_PERSIST for this request.")

    @model_validator(mode="after")
    def check_lengths(self) -> "EvaluateRequest":
        if len(self.prompts) != len(self.responses):
            raise ValueError("prompts and responses must have the same length")
        return self


class BatchEvaluateRequest(BaseModel):
    """Multiple sessions in one call."""
    sessions: list[EvaluateRequest] = Field(..., min_length=1)


class TurnResultSchema(BaseModel):
    turn_index:       int
    verdict:          str
    is_breach:        bool
    is_benign:        bool
    latency_ms:       float
    stage_reached:    int
    refusal_detected: bool
    labels:           dict[str, Optional[bool]]
    raw_judge_output: str


class SessionSummarySchema(BaseModel):
    session_id:    str
    attack_method: str
    target_model:  str
    behavior_id:   str
    status:        str
    total_turns:   int
    breach_turn:   Optional[int]
    ttb:           Optional[int]
    breached:      bool
    turns:         list[TurnResultSchema]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_evaluator() -> MultiTurnEvaluator:
    if _evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialised.")
    return _evaluator


def _summary_to_response(summary: SessionSummary) -> dict[str, Any]:
    return summary.to_dict()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    """
    Liveness / readiness probe.

    Returns judge load status, DB connectivity, and uptime info.
    """
    ev = _get_evaluator()
    db_ok = ev._store is not None
    return {
        "status":       "ok",
        "judge_loaded": ev._judge._judge.loaded,
        "db_connected": db_ok,
        "lazy_load":    _JUDGE_LAZY_LOAD,
        "max_batch":    _JUDGE_MAX_BATCH,
        "persist":      _MONGO_PERSIST,
    }


@app.post("/evaluate/session", response_model=SessionSummarySchema, tags=["Evaluation"])
async def evaluate_session(req: EvaluateRequest) -> dict[str, Any]:
    """
    Evaluate a single multi-turn session.

    Runs the full pipeline synchronously (in a thread executor so the event
    loop is not blocked) and returns the complete SessionSummary when done.

    Use ``/ws/evaluate`` for turn-by-turn streaming to a frontend.

    **Request body** — see ``EvaluateRequest`` schema.

    **Response** — ``SessionSummarySchema`` with per-turn verdicts.
    """
    ev   = _get_evaluator()
    loop = asyncio.get_event_loop()

    # Optionally override persistence for this request
    original_persist = ev._persist
    if req.persist is not None:
        ev._persist = req.persist

    try:
        summary: SessionSummary = await loop.run_in_executor(
            None,
            lambda: ev.run_session(
                prompts=req.prompts,
                responses=req.responses,
                attack_method=req.attack_method,
                target_model=req.target_model,
                behavior_id=req.behavior_id,
                session_id=req.session_id,
                benign_turns=set(req.benign_turns),
            ),
        )
    finally:
        ev._persist = original_persist

    return _summary_to_response(summary)


@app.post("/evaluate/batch", tags=["Evaluation"])
async def evaluate_batch(req: BatchEvaluateRequest) -> dict[str, Any]:
    """
    Evaluate multiple sessions in sequence.

    Each session in ``sessions`` follows the same schema as ``/evaluate/session``.
    Results are returned in the same order as the input.

    Note: sessions share a single GPU; they run sequentially. For concurrent
    multi-client usage prefer the WebSocket endpoint.
    """
    ev      = _get_evaluator()
    loop    = asyncio.get_event_loop()

    async def _run_one(req_item: EvaluateRequest) -> dict[str, Any]:
        summary = await loop.run_in_executor(
            None,
            lambda: ev.run_session(
                prompts=req_item.prompts,
                responses=req_item.responses,
                attack_method=req_item.attack_method,
                target_model=req_item.target_model,
                behavior_id=req_item.behavior_id,
                session_id=req_item.session_id,
                benign_turns=set(req_item.benign_turns),
            ),
        )
        return _summary_to_response(summary)

    results = []
    for r in req.sessions:
        results.append(await _run_one(r))

    return {"sessions": results, "total": len(results)}


@app.get("/report", tags=["Analytics"])
async def get_report(
    attack_method: Optional[str] = Query(None, description="Filter by attack method."),
    target_model:  Optional[str] = Query(None, description="Filter by target model."),
) -> dict[str, Any]:
    """
    Generate and return the full metrics report from MongoDB.

    Returns ASR, TTB, Refusal Robustness, and the vulnerability heatmap
    data as a JSON object. Requires a connected MongoDB instance.

    Query params
    ------------
    attack_method : Filter results to a specific attack type.
    target_model  : Filter results to a specific model.
    """
    ev = _get_evaluator()
    if not ev._store:
        raise HTTPException(status_code=503, detail="MongoDB not connected — report unavailable.")

    loop   = asyncio.get_event_loop()
    report = await loop.run_in_executor(
        None,
        lambda: ev.generate_report(
            attack_method=attack_method,
            target_model=target_model,
            save_dir=None,
            show_plots=False,
        ),
    )
    if report is None:
        raise HTTPException(status_code=500, detail="Report generation failed.")
    return report


@app.get("/sessions", tags=["Sessions"])
async def list_sessions(
    attack_method: Optional[str] = Query(None),
    target_model:  Optional[str] = Query(None),
    status:        Optional[str] = Query(None, description="BREACHED | EXHAUSTED | IN_PROGRESS"),
    limit:         int           = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """
    List stored evaluation sessions from MongoDB.

    Query params
    ------------
    attack_method : Filter by attack method name.
    target_model  : Filter by model name.
    status        : Filter by session status (BREACHED / EXHAUSTED / IN_PROGRESS).
    limit         : Maximum number of sessions to return (1–500, default 50).
    """
    ev = _get_evaluator()
    if not ev._store:
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    filters: dict[str, Any] = {}
    if attack_method:
        filters["attack_method"] = attack_method
    if target_model:
        filters["target_model"] = target_model
    if status:
        filters["status"] = status

    loop     = asyncio.get_event_loop()
    sessions = await loop.run_in_executor(
        None, lambda: ev._store.get_all_sessions(filters or None)
    )
    sessions = sessions[:limit]
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str) -> dict[str, Any]:
    """
    Retrieve a single session document by its UUID.

    Returns the session metadata. Per-turn details are stored in the ``turns``
    collection; use ``/report`` for aggregated metrics.
    """
    ev = _get_evaluator()
    if not ev._store:
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    loop = asyncio.get_event_loop()
    doc  = await loop.run_in_executor(None, ev._store.get_session, session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return doc


# ---------------------------------------------------------------------------
# WebSocket endpoint — streaming evaluation
# ---------------------------------------------------------------------------

@app.websocket("/ws/evaluate")
async def ws_evaluate(ws: WebSocket) -> None:
    """
    WebSocket endpoint for real-time streaming evaluation.

    Protocol
    --------
    1. Client connects.
    2. Client sends one JSON message matching ``EvaluateRequest``.
    3. Server streams one ``turn`` message per evaluated turn:
           {"type": "turn", "data": <TurnResultSchema>}
    4. Server sends a final ``summary`` message:
           {"type": "summary", "data": <SessionSummarySchema>}
    5. Connection closes.

    On validation or runtime errors the server sends:
           {"type": "error", "message": "<description>"}
    and closes the connection.

    Concurrency
    -----------
    Simultaneous connections are limited to MAX_CONCURRENT_WS (default 8).
    Additional clients receive a ``{"type": "error", "message": "Server busy"}``
    message immediately.
    """
    await ws.accept()

    if not _WS_SEMAPHORE.locked() or _WS_SEMAPHORE._value > 0:
        pass  # Semaphore will be acquired below
    else:
        await ws.send_json({"type": "error", "message": "Server busy — try again shortly."})
        await ws.close(code=1013)
        return

    async with _WS_SEMAPHORE:
        try:
            raw = await ws.receive_text()
            data = json.loads(raw)
            req  = EvaluateRequest(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            await ws.send_json({"type": "error", "message": f"Invalid request: {exc}"})
            await ws.close(code=1003)
            return

        ev = _get_evaluator()

        async def _stream_turn(tr: TurnResult) -> None:
            await ws.send_json({"type": "turn", "data": tr.to_dict()})

        try:
            summary = await ev.run_session_streaming(
                prompts=req.prompts,
                responses=req.responses,
                on_turn=_stream_turn,
                attack_method=req.attack_method,
                target_model=req.target_model,
                behavior_id=req.behavior_id,
                session_id=req.session_id,
                benign_turns=set(req.benign_turns),
            )
            await ws.send_json({"type": "summary", "data": _summary_to_response(summary)})
        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected mid-session.")
        except Exception as exc:
            logger.error("[WS] Session error: %s", exc, exc_info=True)
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,        # Must be 1 — GPU model is a singleton
        log_level="info",
        reload=False,
    )
