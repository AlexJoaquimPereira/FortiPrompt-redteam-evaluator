"""
evaluator.py — Multi-Turn Session Evaluator
============================================
Orchestrates the full pipeline:

  Step 0  OpenAI log normalisation  (openai_schema.py)
  Step 1  Fast-Refusal Filter       (judge_ensemble.py)
  Step 2  WildGuard Unified Inference
  Step 3  Boolean Classification → Breach detection

Persistent storage in MongoDB via SessionStore with stop-and-resume support.

Multi-turn logic
----------------
  Turn 0  : Benign seed — expected SAFE, tagged is_benign=True in DB.
  Turn N  : Payload — full 3-step pipeline.
  Breach  : SUCCESS verdict → session flagged BREACHED, TTB logged.
  Resume  : already_evaluated() guard skips turns already in the DB.

Integration with FortiPrompt
-----------------------------
Two high-level entry points:

  run_session(turns, ...)          — list of (prompt, response) string pairs
  run_session_openai(payloads, …)  — list of raw OpenAI ChatCompletion dicts

Both return a SessionSummary with per-turn EnsembleResult objects.
For single-shot evaluation without a session use judge_ensemble.evaluate_response().
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from judge_ensemble import EnsembleJudge, EnsembleResult, JudgeVerdict
from openai_schema import OpenAIContextExtractor
from session_store import SessionStore
from metrics_engine import MetricsEngine, print_full_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    """Returned by every run_session* call."""
    session_id:    str
    attack_method: str
    target_model:  str
    behavior_id:   str
    status:        str             # IN_PROGRESS | BREACHED | EXHAUSTED
    total_turns:   int
    breach_turn:   Optional[int]
    turn_results:  list[EnsembleResult] = field(default_factory=list)

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
# Multi-Turn Evaluator
# ---------------------------------------------------------------------------

class MultiTurnEvaluator:
    """
    High-level driver for multi-turn red-teaming session evaluation.

    Parameters
    ----------
    mongo_uri  : MongoDB connection URI.  Reads HARMBENCH_MONGO_URI env if None.
    judge      : Pre-constructed EnsembleJudge.  Created with defaults if None.
    max_turns  : Hard cap on turns evaluated per session.
    """

    def __init__(
        self,
        mongo_uri:  Optional[str]           = None,
        judge:      Optional[EnsembleJudge] = None,
        max_turns:  int                     = 10,
    ):
        self._store     = SessionStore(mongo_uri=mongo_uri)
        self._judge     = judge or EnsembleJudge()
        self._max_turns = max_turns
        self._extractor = OpenAIContextExtractor()

    # ------------------------------------------------------------------
    # String-based session API
    # ------------------------------------------------------------------

    def run_session(
        self,
        turns:         list[tuple[str, str]],
        attack_method: str            = "",
        target_model:  str            = "",
        behavior_id:   str            = "",
        session_id:    Optional[str]  = None,
        benign_turns:  Optional[set[int]] = None,
    ) -> SessionSummary:
        """
        Evaluate a list of plain (prompt, response) string pairs.

        Parameters
        ----------
        turns         : Ordered list of (attacker_prompt, target_model_response).
        benign_turns  : Set of 0-based turn indices considered benign probes.
                        Defaults to {0} (the seed turn).  These are stored with
                        is_benign=True so compute_refusal_robustness() works.
        """
        sid        = session_id or str(uuid.uuid4())
        benign_set = benign_turns if benign_turns is not None else {0}

        self._store.upsert_session(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
        )

        results:     list[EnsembleResult] = []
        breach_turn: Optional[int]        = None

        for turn_idx, (prompt, response) in enumerate(turns[:self._max_turns]):
            if self._store.already_evaluated(sid, turn_idx):
                logger.info("[Session %s] Turn %d already in DB — skipping.", sid, turn_idx)
                continue

            is_benign = turn_idx in benign_set
            logger.info(
                "[Session %s] Evaluating turn %d/%d%s …",
                sid, turn_idx, len(turns) - 1,
                " (benign)" if is_benign else "",
            )

            result = self._judge.evaluate(
                malicious_prompt=prompt,
                model_response=response,
                session_id=sid,
                turn_index=turn_idx,
            )
            results.append(result)
            self._store.record_turn(result, prompt, response, is_benign=is_benign)

            logger.info(
                "[Session %s] Turn %d → %-14s (%.0f ms)  I=%s H=%s R=%s",
                sid, turn_idx, result.verdict.value, result.latency_ms,
                result.labels.intent_harm    if result.labels else "?",
                result.labels.response_harm  if result.labels else "?",
                result.labels.refusal_signal if result.labels else "?",
            )

            if result.verdict == JudgeVerdict.SUCCESS:
                breach_turn = turn_idx
                self._store.mark_breached(sid, breach_turn)
                logger.info(
                    "[Session %s] *** BREACH at turn %d (TTB=%d) ***",
                    sid, turn_idx, turn_idx,
                )
                break
        else:
            self._store.mark_exhausted(sid, total_turns=len(results))

        session_doc = self._store.get_session(sid) or {}
        return SessionSummary(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
            status=session_doc.get("status", "UNKNOWN"),
            total_turns=len(results),
            breach_turn=breach_turn,
            turn_results=results,
        )

    # ------------------------------------------------------------------
    # OpenAI-native session API
    # ------------------------------------------------------------------

    def run_session_openai(
        self,
        payloads:      list[dict[str, Any]],
        attack_method: str            = "",
        target_model:  str            = "",
        behavior_id:   str            = "",
        session_id:    Optional[str]  = None,
        benign_turns:  Optional[set[int]] = None,
    ) -> SessionSummary:
        """
        Evaluate a list of raw OpenAI ChatCompletion / ChatCompletionChunk dicts.
        Step 0 (log normalisation) runs automatically inside the judge.

        Each dict in payloads represents one conversation turn.  The extractor
        flattens its messages array into (user_intent, target_output) before
        passing to WildGuard, so multi-turn context is preserved.

        Parameters
        ----------
        payloads      : One ChatCompletion dict per turn (e.g. from production logs).
        benign_turns  : Turn indices to mark as benign probes (default: {0}).
        """
        sid        = session_id or str(uuid.uuid4())
        benign_set = benign_turns if benign_turns is not None else {0}

        self._store.upsert_session(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
        )

        results:     list[EnsembleResult] = []
        breach_turn: Optional[int]        = None

        for turn_idx, payload in enumerate(payloads[:self._max_turns]):
            if self._store.already_evaluated(sid, turn_idx):
                logger.info("[Session %s] Turn %d already in DB — skipping.", sid, turn_idx)
                continue

            is_benign = turn_idx in benign_set
            logger.info(
                "[Session %s] (OpenAI) Evaluating turn %d/%d%s …",
                sid, turn_idx, len(payloads) - 1,
                " (benign)" if is_benign else "",
            )

            result = self._judge.evaluate_openai(
                payload=payload,
                session_id=sid,
                turn_index=turn_idx,
            )
            results.append(result)

            # Extract (prompt, response) strings for DB storage.
            # We run extraction a second time here rather than inside the judge
            # so the stored strings always mirror what the judge actually received.
            exchange = self._extractor.extract(payload)
            self._store.record_turn(
                result,
                exchange.user_intent,
                exchange.target_output,
                is_benign=is_benign,
            )

            logger.info(
                "[Session %s] Turn %d → %-14s (%.0f ms)  tool_calls=%s  I=%s H=%s R=%s",
                sid, turn_idx, result.verdict.value, result.latency_ms,
                result.tool_calls_found,
                result.labels.intent_harm    if result.labels else "?",
                result.labels.response_harm  if result.labels else "?",
                result.labels.refusal_signal if result.labels else "?",
            )

            if result.verdict == JudgeVerdict.SUCCESS:
                breach_turn = turn_idx
                self._store.mark_breached(sid, breach_turn)
                logger.info(
                    "[Session %s] *** BREACH at turn %d (TTB=%d) ***",
                    sid, turn_idx, turn_idx,
                )
                break
        else:
            self._store.mark_exhausted(sid, total_turns=len(results))

        session_doc = self._store.get_session(sid) or {}
        return SessionSummary(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
            status=session_doc.get("status", "UNKNOWN"),
            total_turns=len(results),
            breach_turn=breach_turn,
            turn_results=results,
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

    def run_batch_openai(self, sessions: list[dict[str, Any]]) -> list[SessionSummary]:
        """
        Evaluate multiple OpenAI-format sessions sequentially.
        Each dict maps to the kwargs of run_session_openai().
        """
        summaries: list[SessionSummary] = []
        for i, sess in enumerate(sessions):
            logger.info("Batch (OpenAI): session %d/%d …", i + 1, len(sessions))
            summaries.append(self.run_session_openai(**sess))
        return summaries

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def generate_report(
        self,
        attack_method: Optional[str]        = None,
        target_model:  Optional[str]        = None,
        save_dir:      Optional[str | Path] = None,
        show_plots:    bool                 = True,
    ) -> None:
        """Compute all four metrics and print/plot them."""
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
            plot_vulnerability_heatmap(
                heatmap,
                save_path=save_dir / "vulnerability_heatmap.png",
                show=show_plots,
            )
            plot_ttb_distribution(
                ttb,
                save_path=save_dir / "ttb_distribution.png",
                show=show_plots,
            )
            plot_label_breakdown(
                asr,
                save_path=save_dir / "label_breakdown.png",
                show=show_plots,
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release VRAM and close the MongoDB connection."""
        self._judge.teardown()
        self._store.close()

    def __enter__(self) -> "MultiTurnEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()