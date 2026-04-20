"""
evaluator.py — Multi-Turn Session Evaluator
============================================
Orchestrates the full pipeline:

  Step 1  Fast-Refusal Filter       (judge_ensemble.py)
  Step 2  WildGuard Unified Inference
  Step 3  Boolean Classification → Breach detection

Input contract
--------------
All public methods accept **plain string lists**:

    prompts   : list[str]   — ordered attacker / user turns
    responses : list[str]   — ordered target model responses (same length)

These two lists are zipped into (prompt, response) pairs internally.
The OpenAI ChatCompletion layer has been removed entirely.

Persistent storage
------------------
MongoDB writes happen in a background asyncio task so they never block
the streaming WebSocket response.  Pass ``persist=False`` to disable.

Throughput optimisations
------------------------
  • Entire session batched into one EnsembleJudge.evaluate_batch() call
    which packs non-regex turns into a single padded GPU forward pass.
  • DB writes are fire-and-forget (asyncio.ensure_future).
  • run_batch_async() evaluates multiple sessions concurrently using
    asyncio.gather, limited by a semaphore so the GPU isn't over-subscribed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from judge_ensemble import EnsembleJudge, EnsembleResult, JudgeVerdict
from session_store import SessionStore
from metrics_engine import MetricsEngine, print_full_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Result for a single evaluated turn, ready to stream to the UI."""
    turn_index:       int
    prompt:           str
    response:         str
    verdict:          str            # JudgeVerdict.value
    is_breach:        bool
    is_benign:        bool
    latency_ms:       float
    stage_reached:    int
    refusal_detected: bool
    intent_harm:      Optional[bool]
    response_harm:    Optional[bool]
    refusal_signal:   Optional[bool]
    raw_judge_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index":       self.turn_index,
            "verdict":          self.verdict,
            "is_breach":        self.is_breach,
            "is_benign":        self.is_benign,
            "latency_ms":       round(self.latency_ms, 2),
            "stage_reached":    self.stage_reached,
            "refusal_detected": self.refusal_detected,
            "labels": {
                "intent_harm":    self.intent_harm,
                "response_harm":  self.response_harm,
                "refusal_signal": self.refusal_signal,
            },
            "raw_judge_output": self.raw_judge_output,
        }


@dataclass
class SessionSummary:
    """Returned by every run_session* call. Also sent as the final WS message."""
    session_id:    str
    attack_method: str
    target_model:  str
    behavior_id:   str
    status:        str              # IN_PROGRESS | BREACHED | EXHAUSTED
    total_turns:   int
    breach_turn:   Optional[int]
    turn_results:  list[TurnResult] = field(default_factory=list)

    @property
    def ttb(self) -> Optional[int]:
        """Turns-to-Breach — index of the first SUCCESS turn, or None."""
        return self.breach_turn

    @property
    def breached(self) -> bool:
        return self.status == "BREACHED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":    self.session_id,
            "attack_method": self.attack_method,
            "target_model":  self.target_model,
            "behavior_id":   self.behavior_id,
            "status":        self.status,
            "total_turns":   self.total_turns,
            "breach_turn":   self.breach_turn,
            "ttb":           self.ttb,
            "breached":      self.breached,
            "turns":         [r.to_dict() for r in self.turn_results],
        }


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class MultiTurnEvaluator:
    """
    High-level driver for multi-turn red-teaming session evaluation.

    Parameters
    ----------
    mongo_uri  : MongoDB connection URI. Reads HARMBENCH_MONGO_URI env if None.
                 Pass an empty string "" or set persist=False to disable DB.
    judge      : Pre-constructed EnsembleJudge (kept resident for throughput).
    max_turns  : Hard cap on turns evaluated per session.
    persist    : Whether to write results to MongoDB. Default True.
    """

    def __init__(
        self,
        mongo_uri:  Optional[str]           = None,
        judge:      Optional[EnsembleJudge] = None,
        max_turns:  int                     = 10,
        persist:    bool                    = True,
    ):
        self._judge     = judge or EnsembleJudge(lazy_load=False)
        self._max_turns = max_turns
        self._persist   = persist
        self._store: Optional[SessionStore] = None

        if persist:
            try:
                self._store = SessionStore(mongo_uri=mongo_uri)
            except Exception as exc:
                logger.warning(
                    "[DB] Could not connect to MongoDB (%s). "
                    "Results will NOT be persisted.", exc
                )
                self._store = None

    # ------------------------------------------------------------------
    # Primary synchronous API
    # ------------------------------------------------------------------

    def run_session(
        self,
        prompts:       list[str],
        responses:     list[str],
        attack_method: str            = "",
        target_model:  str            = "",
        behavior_id:   str            = "",
        session_id:    Optional[str]  = None,
        benign_turns:  Optional[set[int]] = None,
    ) -> SessionSummary:
        """
        Evaluate a session expressed as two parallel lists.

        Parameters
        ----------
        prompts       : Ordered attacker / user turns.
        responses     : Ordered target model responses (same length as prompts).
        benign_turns  : 0-based indices of benign probe turns (default: {0}).
                        Stored with is_benign=True for Refusal Robustness metric.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")

        sid        = session_id or str(uuid.uuid4())
        benign_set = benign_turns if benign_turns is not None else {0}
        n          = min(len(prompts), self._max_turns)

        if self._store:
            self._store.upsert_session(
                session_id=sid,
                attack_method=attack_method,
                target_model=target_model,
                behavior_id=behavior_id,
            )

        # ── Determine which turns need evaluation (resume support) ────
        to_evaluate: list[int] = []
        for i in range(n):
            if self._store and self._store.already_evaluated(sid, i):
                logger.info("[Session %s] Turn %d already in DB — skipping.", sid, i)
            else:
                to_evaluate.append(i)

        # ── Batch-evaluate all pending turns in one GPU pass ─────────
        batch_prompts   = [prompts[i]   for i in to_evaluate]
        batch_responses = [responses[i] for i in to_evaluate]
        raw_results: list[EnsembleResult] = []

        if batch_prompts:
            raw_results = self._judge.evaluate_batch(
                prompts=batch_prompts,
                responses=batch_responses,
                session_id=sid,
                start_turn_index=to_evaluate[0] if to_evaluate else 0,
            )
            # Fix turn indices (start_turn_index only works for contiguous ranges)
            for j, res in enumerate(raw_results):
                res.turn_index = to_evaluate[j]

        # ── Assemble ordered TurnResult list ──────────────────────────
        result_map: dict[int, EnsembleResult] = {
            to_evaluate[j]: raw_results[j] for j in range(len(raw_results))
        }

        turn_results: list[TurnResult] = []
        breach_turn:  Optional[int]    = None

        for i in range(n):
            is_benign = i in benign_set
            res = result_map.get(i)
            if res is None:
                continue   # Already evaluated in a prior run; skip for this summary

            tr = TurnResult(
                turn_index=i,
                prompt=prompts[i],
                response=responses[i],
                verdict=res.verdict.value,
                is_breach=(res.verdict == JudgeVerdict.SUCCESS),
                is_benign=is_benign,
                latency_ms=res.latency_ms,
                stage_reached=res.stage_reached,
                refusal_detected=res.refusal_detected,
                intent_harm=res.labels.intent_harm    if res.labels else None,
                response_harm=res.labels.response_harm  if res.labels else None,
                refusal_signal=res.labels.refusal_signal if res.labels else None,
                raw_judge_output=res.raw_judge_output,
            )
            turn_results.append(tr)

            if self._store:
                self._store.record_turn(res, prompts[i], responses[i], is_benign=is_benign)

            logger.info(
                "[Session %s] Turn %d → %-14s (%.0f ms)  I=%s H=%s R=%s",
                sid, i, res.verdict.value, res.latency_ms,
                res.labels.intent_harm    if res.labels else "?",
                res.labels.response_harm  if res.labels else "?",
                res.labels.refusal_signal if res.labels else "?",
            )

            if res.verdict == JudgeVerdict.SUCCESS and breach_turn is None:
                breach_turn = i
                logger.info("[Session %s] *** BREACH at turn %d ***", sid, i)

        # ── Mark session terminal state ───────────────────────────────
        if self._store:
            if breach_turn is not None:
                self._store.mark_breached(sid, breach_turn)
            else:
                self._store.mark_exhausted(sid, total_turns=len(turn_results))

        status = "BREACHED" if breach_turn is not None else "EXHAUSTED"
        return SessionSummary(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
            status=status,
            total_turns=len(turn_results),
            breach_turn=breach_turn,
            turn_results=turn_results,
        )

    # ------------------------------------------------------------------
    # Async streaming API  (used by the FastAPI WebSocket endpoint)
    # ------------------------------------------------------------------

    async def run_session_streaming(
        self,
        prompts:       list[str],
        responses:     list[str],
        on_turn:       Callable[[TurnResult], Any],
        attack_method: str            = "",
        target_model:  str            = "",
        behavior_id:   str            = "",
        session_id:    Optional[str]  = None,
        benign_turns:  Optional[set[int]] = None,
    ) -> SessionSummary:
        """
        Async variant that calls ``on_turn(TurnResult)`` after each turn
        is evaluated, enabling real-time WebSocket streaming.

        The GPU work runs in a thread-pool executor so the event loop
        remains free to serve other connections during inference.

        Parameters
        ----------
        on_turn : Awaitable or plain callback invoked per turn result.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")

        sid        = session_id or str(uuid.uuid4())
        benign_set = benign_turns if benign_turns is not None else {0}
        n          = min(len(prompts), self._max_turns)
        loop       = asyncio.get_event_loop()

        if self._store:
            self._store.upsert_session(
                session_id=sid,
                attack_method=attack_method,
                target_model=target_model,
                behavior_id=behavior_id,
            )

        turn_results: list[TurnResult] = []
        breach_turn:  Optional[int]    = None

        for i in range(n):
            is_benign = i in benign_set

            if self._store and self._store.already_evaluated(sid, i):
                logger.info("[Session %s] Turn %d already in DB — skipping.", sid, i)
                continue

            # Run GPU inference in executor (non-blocking)
            res: EnsembleResult = await loop.run_in_executor(
                None,
                self._judge.evaluate,
                prompts[i],
                responses[i],
                sid,
                i,
            )

            tr = TurnResult(
                turn_index=i,
                prompt=prompts[i],
                response=responses[i],
                verdict=res.verdict.value,
                is_breach=(res.verdict == JudgeVerdict.SUCCESS),
                is_benign=is_benign,
                latency_ms=res.latency_ms,
                stage_reached=res.stage_reached,
                refusal_detected=res.refusal_detected,
                intent_harm=res.labels.intent_harm    if res.labels else None,
                response_harm=res.labels.response_harm  if res.labels else None,
                refusal_signal=res.labels.refusal_signal if res.labels else None,
                raw_judge_output=res.raw_judge_output,
            )
            turn_results.append(tr)

            # Stream turn to client
            if asyncio.iscoroutinefunction(on_turn):
                await on_turn(tr)
            else:
                on_turn(tr)

            # Persist in background (fire-and-forget)
            if self._store:
                asyncio.ensure_future(
                    loop.run_in_executor(
                        None, self._store.record_turn,
                        res, prompts[i], responses[i], is_benign,
                    )
                )

            logger.info(
                "[Session %s] Turn %d → %-14s (%.0f ms)",
                sid, i, res.verdict.value, res.latency_ms,
            )

            if res.verdict == JudgeVerdict.SUCCESS and breach_turn is None:
                breach_turn = i
                logger.info("[Session %s] *** BREACH at turn %d ***", sid, i)

        # Mark session terminal state in background
        if self._store:
            if breach_turn is not None:
                asyncio.ensure_future(
                    loop.run_in_executor(None, self._store.mark_breached, sid, breach_turn)
                )
            else:
                asyncio.ensure_future(
                    loop.run_in_executor(
                        None, self._store.mark_exhausted, sid, len(turn_results)
                    )
                )

        status = "BREACHED" if breach_turn is not None else "EXHAUSTED"
        return SessionSummary(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
            status=status,
            total_turns=len(turn_results),
            breach_turn=breach_turn,
            turn_results=turn_results,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def run_batch(self, sessions: list[dict[str, Any]]) -> list[SessionSummary]:
        """
        Evaluate multiple sessions sequentially.
        Each dict maps to the kwargs of run_session().
        """
        summaries: list[SessionSummary] = []
        for i, sess in enumerate(sessions):
            logger.info("Batch: session %d/%d …", i + 1, len(sessions))
            summaries.append(self.run_session(**sess))
        return summaries

    async def run_batch_async(
        self,
        sessions:     list[dict[str, Any]],
        concurrency:  int = 4,
    ) -> list[SessionSummary]:
        """
        Evaluate multiple sessions with bounded concurrency.
        Useful when sessions are CPU/IO bound; GPU inference is serialised
        internally by the shared EnsembleJudge.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(sess: dict) -> SessionSummary:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.run_session, **sess)

        return list(await asyncio.gather(*[_run_one(s) for s in sessions]))

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def generate_report(
        self,
        attack_method: Optional[str]        = None,
        target_model:  Optional[str]        = None,
        save_dir:      Optional[str | Path] = None,
        show_plots:    bool                 = True,
    ) -> Optional[dict[str, Any]]:
        """Compute all four metrics and optionally print/plot them.

        Returns a dict with ASR, TTB, RR, and heatmap data suitable for
        serialisation to JSON (used by the /report REST endpoint).
        """
        if not self._store:
            logger.warning("No store connected; skipping report.")
            return None

        filters: dict[str, Any] = {}
        if attack_method:
            filters["attack_method"] = attack_method
        if target_model:
            filters["target_model"] = target_model

        turns           = self._store.all_turns(filters=filters if filters else None)
        breach_sessions = self._store.breach_turns()

        engine  = MetricsEngine(turns)
        asr     = engine.compute_asr()
        ttb     = engine.compute_ttb(breach_sessions)
        rr      = engine.compute_refusal_robustness()
        heatmap = engine.compute_heatmap()

        print_full_report(
            asr=asr, ttb=ttb, rr=rr, heatmap=heatmap,
            model_name=target_model or "all",
            method=attack_method or "all",
        )

        if save_dir:
            from metrics_engine import (
                plot_vulnerability_heatmap,
                plot_ttb_distribution,
                plot_label_breakdown,
            )
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_vulnerability_heatmap(heatmap, save_path=save_dir / "vulnerability_heatmap.png", show=show_plots)
            plot_ttb_distribution(ttb,          save_path=save_dir / "ttb_distribution.png",       show=show_plots)
            plot_label_breakdown(asr,            save_path=save_dir / "label_breakdown.png",        show=show_plots)

        return {
            "asr": {
                "total_samples":          asr.total_samples,
                "successes":              asr.successes,
                "failures":               asr.failures,
                "hard_refusals":          asr.hard_refusals,
                "errors":                 asr.errors,
                "asr":                    asr.asr,
                "asr_pct":                asr.asr_pct,
                "intent_harmful_count":   asr.intent_harmful_count,
                "response_harmful_count": asr.response_harmful_count,
                "refusal_count":          asr.refusal_count,
            },
            "ttb": {
                "sessions_evaluated": ttb.sessions_evaluated,
                "sessions_breached":  ttb.sessions_breached,
                "mean_ttb":           ttb.mean_ttb,
                "median_ttb":         ttb.median_ttb,
                "min_ttb":            ttb.min_ttb,
                "max_ttb":            ttb.max_ttb,
                "ttb_distribution":   ttb.ttb_distribution,
            },
            "refusal_robustness": {
                "benign_total":       rr.benign_total,
                "false_refusals":     rr.false_refusals,
                "refusal_robustness": rr.refusal_robustness,
            },
            "heatmap": {
                "categories":     heatmap.categories,
                "attack_methods": heatmap.attack_methods,
                "matrix":         heatmap.matrix,
            },
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._judge.teardown()
        if self._store:
            self._store.close()

    def __enter__(self) -> "MultiTurnEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
