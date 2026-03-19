"""
evaluator.py — Multi-Turn Session Evaluator
============================================
Ties together the EnsembleJudge, SessionStore, and MetricsEngine into a
single high-level API that follows the HarmBench multi-turn logic:

  Turn 1  : Attacker sends a "benign" seed  → Judge marks SAFE.
  Turn N  : Attacker sends the payload      → Full 3-stage pipeline.
  Breach  : If Stage 3 returns YES, session is flagged SUCCESS and TTB is logged.
  Resume  : Already-evaluated (session, turn) pairs are skipped automatically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from judge_ensemble import EnsembleJudge, EnsembleResult, JudgeVerdict
from session_store import SessionStore
from metrics_engine import MetricsEngine, print_full_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data class — one per session
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    session_id:     str
    attack_method:  str
    target_model:   str
    behavior_id:    str
    status:         str          # IN_PROGRESS | BREACHED | EXHAUSTED
    total_turns:    int
    breach_turn:    Optional[int]
    turn_results:   list[EnsembleResult] = field(default_factory=list)

    @property
    def ttb(self) -> Optional[int]:
        return self.breach_turn


# ---------------------------------------------------------------------------
# Multi-Turn Evaluator
# ---------------------------------------------------------------------------

class MultiTurnEvaluator:
    """
    High-level driver that manages one or more red-teaming sessions.

    Parameters
    ----------
    db_path       : Path to the SQLite database file (created if absent).
    judge         : A pre-constructed EnsembleJudge instance; one will be
                    created with default settings if not provided.
    max_turns     : Maximum conversation turns before a session is exhausted.
    """

    def __init__(
        self,
        db_path:   str | Path = "harmbench_results.db",
        judge:     Optional[EnsembleJudge] = None,
        max_turns: int = 10,
    ):
        self._store     = SessionStore(db_path)
        self._judge     = judge or EnsembleJudge()
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Session-level API
    # ------------------------------------------------------------------

    def run_session(
        self,
        turns:         list[tuple[str, str]],    # [(malicious_prompt, model_response), ...]
        attack_method: str = "",
        target_model:  str = "",
        behavior_id:   str = "",
        session_id:    Optional[str] = None,
        benign_turns:  set[int] | None = None,   # Turn indices considered benign
    ) -> SessionSummary:
        """
        Evaluate a complete multi-turn conversation.

        Parameters
        ----------
        turns        : Ordered list of (malicious_prompt, model_response) pairs.
                       Turn 0 is typically the benign seed.
        benign_turns : Set of 0-based turn indices treated as benign probes
                       (used for Refusal Robustness metric).
        session_id   : Provide to resume an interrupted session; a new UUID is
                       generated when omitted.

        Returns
        -------
        SessionSummary with the verdict for every evaluated turn.
        """
        sid = session_id or str(uuid.uuid4())
        self._store.upsert_session(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
        )
        benign_set = benign_turns or {0}  # Turn 0 defaults to benign seed

        results:     list[EnsembleResult] = []
        breach_turn: Optional[int]        = None

        for turn_idx, (prompt, response) in enumerate(turns[:self._max_turns]):
            # ── Resume guard ──────────────────────────────────────────
            if self._store.already_evaluated(sid, turn_idx):
                logger.info("[Session %s] Turn %d already in DB — skipping.", sid, turn_idx)
                continue

            logger.info(
                "[Session %s] Evaluating turn %d/%d …",
                sid, turn_idx, len(turns) - 1
            )

            result = self._judge.evaluate(
                malicious_prompt=prompt,
                model_response=response,
                session_id=sid,
                turn_index=turn_idx,
            )
            results.append(result)
            self._store.record_turn(result, prompt, response)

            logger.info(
                "[Session %s] Turn %d → %s  (%.0f ms)",
                sid, turn_idx, result.verdict.value, result.latency_ms
            )

            # ── Breach detection ──────────────────────────────────────
            if result.verdict == JudgeVerdict.SUCCESS:
                breach_turn = turn_idx
                self._store.mark_breached(sid, breach_turn)
                logger.info(
                    "[Session %s] *** BREACH at turn %d (TTB=%d) ***",
                    sid, turn_idx, turn_idx
                )
                break

        else:
            # All turns exhausted without a breach
            self._store.mark_exhausted(sid, total_turns=len(turns))

        session_row = self._store.get_session(sid) or {}
        return SessionSummary(
            session_id=sid,
            attack_method=attack_method,
            target_model=target_model,
            behavior_id=behavior_id,
            status=session_row.get("status", "UNKNOWN"),
            total_turns=len(results),
            breach_turn=breach_turn,
            turn_results=results,
        )

    # ------------------------------------------------------------------
    # Bulk evaluation
    # ------------------------------------------------------------------

    def run_batch(
        self,
        sessions: list[dict[str, Any]],
    ) -> list[SessionSummary]:
        """
        Evaluate a batch of sessions sequentially.

        Each element of ``sessions`` should be a dict compatible with the
        keyword arguments of ``run_session()``:
            {
              "turns":         [(prompt, response), ...],
              "attack_method": "GCG",
              "target_model":  "llama-3-8b",
              "behavior_id":   "behavior_042",
              "session_id":    "optional-resume-id",
            }
        """
        summaries: list[SessionSummary] = []
        for i, sess in enumerate(sessions):
            logger.info("Batch: evaluating session %d/%d …", i + 1, len(sessions))
            summary = self.run_session(**sess)
            summaries.append(summary)
        return summaries

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def generate_report(
        self,
        attack_method: Optional[str] = None,
        target_model:  Optional[str] = None,
        save_dir:      Optional[str | Path] = None,
        show_plots:    bool = True,
    ) -> None:
        """
        Compute all four metrics from the database and print/plot results.

        Parameters
        ----------
        attack_method : Filter to a specific attack method (e.g. "GCG").
        target_model  : Filter to a specific target model.
        save_dir      : Directory to write PNG files; None → no files saved.
        show_plots    : Whether to call plt.show() (False for headless runs).
        """
        filters: dict[str, Any] = {}
        if attack_method:
            filters["attack_method"] = attack_method
        if target_model:
            filters["target_model"] = target_model

        turns          = self._store.all_turns(filters=filters or None)
        breach_sessions = self._store.breach_turns()

        engine = MetricsEngine(turns)
        asr     = engine.compute_asr()
        ttb     = engine.compute_ttb(breach_sessions)
        rr      = engine.compute_refusal_robustness()
        heatmap = engine.compute_heatmap()

        print_full_report(
            asr=asr,
            ttb=ttb,
            rr=rr,
            heatmap=heatmap,
            model_name=target_model or "all",
            method=attack_method or "all",
        )

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            from metrics_engine import plot_vulnerability_heatmap, plot_ttb_distribution
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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources (judge VRAM + DB connection)."""
        self._judge.teardown()
        self._store.close()

    def __enter__(self) -> "MultiTurnEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
