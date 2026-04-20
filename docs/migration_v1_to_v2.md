# FortiPrompt v2 — Migration Guide

## What Changed in v2

### Input contract

**v1** accepted OpenAI `ChatCompletion` JSON objects or `{"messages": [...]}` dicts.

**v2** accepts two plain string lists:
- `prompts: list[str]`   — ordered attacker / user turns
- `responses: list[str]` — ordered target model responses (same length)

`openai_schema.py` has been removed entirely.

---

## Updating Your Code

### Direct Python API

**Before (v1)**
```python
turns = [
    ("user prompt 0", "model response 0"),
    ("user prompt 1", "model response 1"),
]
summary = ev.run_session(turns=turns, attack_method="GCG", ...)
```

**After (v2)**
```python
prompts   = ["user prompt 0", "user prompt 1"]
responses = ["model response 0", "model response 1"]
summary   = ev.run_session(prompts=prompts, responses=responses, attack_method="GCG", ...)
```

---

### OpenAI payload conversion

If your external system still produces `ChatCompletion` JSON, extract the strings before calling FortiPrompt:

```python
def extract_turn(payload: dict) -> tuple[str, str]:
    """Extract the last user prompt and last assistant response from a ChatCompletion dict."""
    messages = payload.get("messages", [])
    prompt   = next((m["content"] for m in reversed(messages) if m["role"] == "user"),  "")
    response = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    return prompt, response

# For a session expressed as a list of ChatCompletion payloads:
pairs     = [extract_turn(p) for p in payloads]
prompts   = [p for p, _ in pairs]
responses = [r for _, r in pairs]

summary = ev.run_session(prompts=prompts, responses=responses, ...)
```

---

### Judge API

**Before (v1)**
```python
result = judge.evaluate(
    malicious_prompt="...",
    model_response="...",
)

# OpenAI payload variant
result = judge.evaluate_openai({"messages": [...]})
```

**After (v2)**
```python
result = judge.evaluate(
    prompt="...",
    response="...",
)

# Batch variant (new — single GPU pass for N pairs)
results = judge.evaluate_batch(
    prompts=["...", "..."],
    responses=["...", "..."],
)
```

---

### REST API

The JSON body field names for single-session evaluation have changed:

**Before (v1)** — accepted `{"turns": [["prompt", "response"], ...]}` or OpenAI payloads.

**After (v2)**
```json
{
  "prompts":   ["prompt 0", "prompt 1"],
  "responses": ["response 0", "response 1"],
  "attack_method": "GCG",
  "target_model":  "llama-3-8b",
  "benign_turns":  [0],
  "session_id":    null
}
```

---

### WebSocket protocol

The WebSocket endpoint URL and protocol are unchanged. Only the request body format has changed (same as REST above).

---

## New Features in v2

- **`POST /evaluate/batch`** — submit multiple sessions in one HTTP call.
- **`evaluate_batch()`** — single GPU forward pass for N pairs (significant throughput gain).
- **`JUDGE_MAX_BATCH` env var** — tune GPU batch size without code changes.
- **`persist` per-request flag** — disable MongoDB writes for individual requests.
- **`GET /sessions`** and **`GET /sessions/{id}`** — new session browsing endpoints.
- **`run_session_streaming()`** — async method with per-turn callback for WebSocket streaming.
- **`generate_report()`** now returns a JSON-serialisable dict (also served at `GET /report`).

---

## Removed in v2

| Removed | Reason |
|---|---|
| `openai_schema.py` | Replaced by plain string lists |
| `evaluator.run_session_openai()` | Use `run_session()` with extracted strings |
| `evaluator.run_batch_openai()` | Use `run_batch()` |
| `judge.evaluate_openai()` | Use `judge.evaluate()` or `judge.evaluate_batch()` |
| `EnsembleResult.tool_calls_found` | No longer relevant without OpenAI schema |
| `run_evaluation.py evaluate-openai` sub-command | Use `evaluate` with pre-extracted strings |
