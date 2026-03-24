"""
openai_schema.py — OpenAI-Standard Input Layer  (Step 0: Log Normalisation)
============================================================================
Accepts any OpenAI-compatible payload and converts it to the two flat strings
that WildGuard needs:

    user_intent   — flattened user/system turns  ("Malicious Intent")
    target_output — the assistant turn(s)         ("Target Output")

Supported input shapes
----------------------
1. Full ChatCompletion   {"choices": [{"message": {...}}], "messages": [...]}
2. Streaming chunks      list of ChatCompletionChunk dicts — accumulated here
3. Plain messages list   {"messages": [...]}
4. Minimal adapter dict  {"prompt": "...", "response": "..."}

Multi-turn strategy
-------------------
All prior assistant turns are folded into the Stateful Transcript so the
judge sees full conversational context.  Only the LAST assistant turn becomes
the exclusive target_output under evaluation.

Tool-call interception
----------------------
tool_calls and legacy function_call payloads are serialised and appended to
target_output so code-injection-via-tool is visible to the judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class NormalisedExchange:
    """Provider-agnostic representation of one evaluated exchange."""
    user_intent:      str               # Judge [PROMPT] input
    target_output:    str               # Judge [RESPONSE] input
    raw_messages:     list[dict[str, Any]] = field(default_factory=list)
    tool_calls_found: bool  = False     # True if tool_calls/function_call detected
    model:            str   = ""        # Model id from the payload
    finish_reason:    str   = ""        # "stop" | "tool_calls" | "length" | ""


# ---------------------------------------------------------------------------
# Streaming chunk accumulator
# ---------------------------------------------------------------------------

class ChunkAccumulator:
    """
    Merges a stream of ChatCompletionChunk dicts into a single synthetic
    ChatCompletion-like dict that _from_messages can parse normally.

    Usage
    -----
        acc = ChunkAccumulator()
        for chunk in stream:
            acc.push(chunk)
        exchange = extractor.extract(acc.finalise())
    """

    def __init__(self) -> None:
        self._model: str = ""
        self._role:  str = "assistant"
        self._content_parts: list[str] = []
        self._tool_calls:    list[dict[str, Any]] = []
        self._finish_reason: str = ""

    def push(self, chunk: dict[str, Any]) -> None:
        """Ingest one streaming chunk."""
        if not self._model and chunk.get("model"):
            self._model = chunk["model"]
        choices = chunk.get("choices", [])
        if not choices:
            return
        delta = choices[0].get("delta", {})
        if choices[0].get("finish_reason"):
            self._finish_reason = choices[0]["finish_reason"]
        if delta.get("role"):
            self._role = delta["role"]
        if delta.get("content"):
            self._content_parts.append(delta["content"])
        # Accumulate streamed tool call arguments
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            while len(self._tool_calls) <= idx:
                self._tool_calls.append({"id": "", "type": "function",
                                          "function": {"name": "", "arguments": ""}})
            fn = tc.get("function", {})
            if fn.get("name"):
                self._tool_calls[idx]["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                self._tool_calls[idx]["function"]["arguments"] += fn["arguments"]
            if tc.get("id"):
                self._tool_calls[idx]["id"] = tc["id"]

    def finalise(self) -> dict[str, Any]:
        """Return a synthetic ChatCompletion-like dict ready for extraction."""
        msg: dict[str, Any] = {
            "role":    self._role,
            "content": "".join(self._content_parts),
        }
        if self._tool_calls:
            msg["tool_calls"] = self._tool_calls
        return {
            "model": self._model,
            "messages": [msg],
            "choices": [{"finish_reason": self._finish_reason, "message": msg}],
        }


# ---------------------------------------------------------------------------
# Context extractor
# ---------------------------------------------------------------------------

class OpenAIContextExtractor:
    """
    Converts an OpenAI-format payload into a NormalisedExchange.

    All four input shapes are handled; see module docstring for details.
    """

    _PROMPT_ROLES:   frozenset[str] = frozenset({"system", "user", "tool", "function"})
    _RESPONSE_ROLES: frozenset[str] = frozenset({"assistant"})

    def extract(self, payload: dict[str, Any]) -> NormalisedExchange:
        # Shape 4 — minimal adapter dict
        if "prompt" in payload and "response" in payload and "messages" not in payload:
            return NormalisedExchange(
                user_intent=str(payload["prompt"]),
                target_output=str(payload["response"]),
                model=payload.get("model", ""),
            )

        # Shape 1 / 2 — full ChatCompletion (has choices)
        if "choices" in payload:
            return self._from_completion(payload)

        # Shape 3 — plain messages list
        if "messages" in payload:
            return self._from_messages(
                messages=payload["messages"],
                model=payload.get("model", ""),
            )

        # Fallback: treat the whole dict as a single user turn
        return NormalisedExchange(
            user_intent=json.dumps(payload),
            target_output="(no assistant response)",
        )

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _from_completion(self, payload: dict[str, Any]) -> NormalisedExchange:
        model         = payload.get("model", "")
        finish_reason = ""
        # Start from any messages already embedded in the request payload
        messages: list[dict[str, Any]] = list(payload.get("messages", []))
        choices = payload.get("choices", [])
        if choices:
            choice        = choices[0]
            finish_reason = choice.get("finish_reason", "")
            # ChatCompletion → .message; ChatCompletionChunk → .delta
            incoming_msg  = choice.get("message") or choice.get("delta") or {}
            if incoming_msg:
                # Only append if this assistant turn isn't already the last message
                # (avoids double-counting when the caller includes the full message list
                # alongside choices, which some provider SDKs do)
                last = messages[-1] if messages else {}
                incoming_content = incoming_msg.get("content", "")
                last_content     = last.get("content", "")
                if not (
                    last.get("role") == "assistant"
                    and last_content == incoming_content
                ):
                    messages.append(incoming_msg)

        return self._from_messages(messages, model=model, finish_reason=finish_reason)

    def _from_messages(
        self,
        messages:      list[dict[str, Any]],
        model:         str = "",
        finish_reason: str = "",
    ) -> NormalisedExchange:
        """
        Flatten messages into (user_intent, target_output).

        The LAST assistant turn → target_output.
        All prior turns (including earlier assistant turns) → user_intent
        as the Stateful Transcript.
        """
        # Identify index of the final assistant turn
        last_assistant_idx: Optional[int] = None
        for i, msg in enumerate(messages):
            if _role(msg) == "assistant":
                last_assistant_idx = i

        prompt_parts:    list[str] = []
        response_parts:  list[str] = []
        tool_calls_found             = False

        for i, msg in enumerate(messages):
            role    = _role(msg)
            content = _content(msg)

            if role == "assistant":
                if i == last_assistant_idx:
                    # The response under evaluation
                    if content:
                        response_parts.append(content)
                    tc_text, found = _extract_tool_calls(msg)
                    if found:
                        tool_calls_found = True
                        response_parts.append(tc_text)
                else:
                    # Earlier assistant turn → fold into prompt transcript
                    if content:
                        prompt_parts.append(f"[Assistant, turn {i}]: {content}")
            else:
                prefix = "" if role == "user" else f"[{role.capitalize()}]: "
                if content:
                    prompt_parts.append(f"{prefix}{content}")

        user_intent   = "\n\n".join(prompt_parts).strip()
        target_output = "\n\n".join(response_parts).strip()

        return NormalisedExchange(
            user_intent=user_intent   or "(no user prompt)",
            target_output=target_output or "(no assistant response)",
            raw_messages=list(messages),
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
    """Extract text content, handling multi-modal list-of-parts format."""
    raw = msg.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
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
    """Serialise tool_calls / function_call into a readable string."""
    parts: list[str] = []

    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "unknown_tool")
            args = fn.get("arguments", "{}")
            parts.append(f"[TOOL_CALL] {name}({args})")

    fc = msg.get("function_call")
    if fc and isinstance(fc, dict):
        name = fc.get("name", "unknown_function")
        args = fc.get("arguments", "{}")
        parts.append(f"[FUNCTION_CALL] {name}({args})")

    if parts:
        return "\n".join(parts), True
    return "", False