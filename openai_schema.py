"""
openai_schema.py — OpenAI-Standard Input Layer
================================================
Step 0 of the pipeline: Log Normalization.

Accepts standard ``ChatCompletion`` or ``ChatCompletionChunk`` JSON objects
(the same format produced by the OpenAI Python SDK, LiteLLM, or any
OpenAI-compatible provider) and converts them into the two flat strings the
WildGuard judge needs:

    • user_intent   – flattened user/system turns  (the "Malicious Intent")
    • target_output – the assistant turn(s)         (the "Target Output")

Multi-turn support
------------------
For multi-turn conversations the ``messages`` array is flattened into a single
"Stateful Transcript" that preserves chronological order so the judge sees
the full context, not just the last exchange.

Tool-call interception
----------------------
Per Step 0 spec: if the assistant turn contains ``tool_calls`` or
``function_call`` (legacy), their serialised arguments are appended to
``target_output`` so the judge can detect code-injection via tool execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Public data class — produced by the extractor
# ---------------------------------------------------------------------------

@dataclass
class NormalisedExchange:
    """
    Provider-agnostic representation of a single evaluated exchange.

    Fields
    ------
    user_intent      : Flattened user/system content used as the judge's
                       [PROMPT] input.  In multi-turn sessions this is the
                       concatenated Stateful Transcript up to (but excluding)
                       the final assistant turn.
    target_output    : The assistant content (+ any tool-call payloads) used
                       as the judge's [RESPONSE] input.
    raw_messages     : Original messages list preserved for traceability.
    tool_calls_found : True if any tool_calls / function_call was detected
                       in the assistant turn.
    model            : Model identifier extracted from the completion object
                       (empty string if not present).
    finish_reason    : "stop" | "tool_calls" | "length" | "" etc.
    """
    user_intent:       str
    target_output:     str
    raw_messages:      list[dict[str, Any]] = field(default_factory=list)
    tool_calls_found:  bool  = False
    model:             str   = ""
    finish_reason:     str   = ""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class OpenAIContextExtractor:
    """
    Converts a raw OpenAI-format JSON payload into a ``NormalisedExchange``.

    Accepted shapes
    ---------------
    1. Full ``ChatCompletion`` object  (has ``choices[].message``)
    2. Streaming ``ChatCompletionChunk`` accumulation  (has ``choices[].delta``)
    3. Plain ``messages`` list  (for offline log replay)
    4. Minimal dict  {``prompt``: str, ``response``: str}  (testing / adapters)
    """

    # Roles whose content feeds into the PROMPT side of the judge
    _PROMPT_ROLES: frozenset[str] = frozenset({"system", "user", "tool", "function"})
    # Roles whose content feeds into the RESPONSE side
    _RESPONSE_ROLES: frozenset[str] = frozenset({"assistant"})

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(self, payload: dict[str, Any]) -> NormalisedExchange:
        """
        Parse *payload* and return a ``NormalisedExchange``.

        Parameters
        ----------
        payload : Any JSON-deserialisable dict that matches one of the
                  accepted shapes documented above.
        """
        # Shape 4 — minimal adapter dict
        if "prompt" in payload and "response" in payload:
            return NormalisedExchange(
                user_intent=str(payload["prompt"]),
                target_output=str(payload["response"]),
                model=payload.get("model", ""),
            )

        # Shape 3 — plain messages list
        if "messages" in payload and "choices" not in payload:
            return self._from_messages(
                messages=payload["messages"],
                model=payload.get("model", ""),
            )

        # Shape 1 & 2 — ChatCompletion / ChatCompletionChunk
        return self._from_completion(payload)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _from_completion(self, payload: dict[str, Any]) -> NormalisedExchange:
        model        = payload.get("model", "")
        choices      = payload.get("choices", [])
        finish_reason = ""

        # Reconstruct messages from the completion object.
        # The caller is expected to pass the *request* messages alongside
        # the completion; if not present we treat the completion content
        # as the only turn.
        messages: list[dict[str, Any]] = list(payload.get("messages", []))

        if choices:
            choice       = choices[0]
            finish_reason = choice.get("finish_reason", "")
            # ChatCompletion → .message;  Chunk → .delta
            msg = choice.get("message") or choice.get("delta") or {}
            if msg:
                messages.append(msg)

        return self._from_messages(messages, model=model, finish_reason=finish_reason)

    def _from_messages(
        self,
        messages:      list[dict[str, Any]],
        model:         str = "",
        finish_reason: str = "",
    ) -> NormalisedExchange:
        """
        Flatten ``messages`` into (user_intent, target_output).

        Multi-turn strategy
        -------------------
        All user/system turns are concatenated (oldest first) to form the
        Stateful Transcript that constitutes ``user_intent``.
        The *last* assistant turn is used as ``target_output``.
        Earlier assistant turns are prepended to ``user_intent`` so the judge
        sees the full conversational context.
        """
        prompt_parts:   list[str] = []
        response_parts: list[str] = []
        tool_calls_found = False

        # We iterate chronologically; only the LAST assistant turn becomes
        # the exclusive response under evaluation.  All prior assistant turns
        # are folded into the transcript so the judge has full context.
        last_assistant_idx = None
        for i, msg in enumerate(messages):
            if _role(msg) == "assistant":
                last_assistant_idx = i

        for i, msg in enumerate(messages):
            role    = _role(msg)
            content = _content(msg)

            if role == "assistant":
                if i == last_assistant_idx:
                    # Primary response under evaluation
                    response_parts.append(content)
                    # --- Tool-call / function-call interception --------
                    tc_text, found = _extract_tool_calls(msg)
                    if found:
                        tool_calls_found = True
                        response_parts.append(tc_text)
                else:
                    # Earlier assistant turns → fold into transcript
                    prompt_parts.append(f"[assistant turn {i}] {content}")
            else:
                # system / user / tool / function
                label = f"[{role}]" if role not in ("user",) else ""
                prompt_parts.append(f"{label} {content}".strip())

        user_intent   = "\n\n".join(p for p in prompt_parts  if p).strip()
        target_output = "\n\n".join(r for r in response_parts if r).strip()

        return NormalisedExchange(
            user_intent=user_intent     or "(no user prompt)",
            target_output=target_output or "(no assistant response)",
            raw_messages=messages,
            tool_calls_found=tool_calls_found,
            model=model,
            finish_reason=finish_reason,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _role(msg: dict[str, Any]) -> str:
    return (msg.get("role") or "user").lower().strip()


def _content(msg: dict[str, Any]) -> str:
    """Extract text content from a message, handling list-of-parts format."""
    raw = msg.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    # OpenAI vision / multi-modal: list of {"type": "text", "text": "..."}
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts).strip()
    return str(raw).strip()


def _extract_tool_calls(msg: dict[str, Any]) -> tuple[str, bool]:
    """
    Return (serialised_tool_payload, found_flag).

    Handles both:
    • ``tool_calls`` list  (current OpenAI spec)
    • ``function_call``    (legacy spec, still used by some providers)
    """
    parts: list[str] = []

    # Current spec
    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "unknown_tool")
            args = fn.get("arguments", "{}")
            parts.append(f"[TOOL_CALL] {name}({args})")

    # Legacy spec
    fc = msg.get("function_call")
    if fc and isinstance(fc, dict):
        name = fc.get("name", "unknown_function")
        args = fc.get("arguments", "{}")
        parts.append(f"[FUNCTION_CALL] {name}({args})")

    if parts:
        return "\n".join(parts), True
    return "", False
