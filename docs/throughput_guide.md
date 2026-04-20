# FortiPrompt — Throughput & Optimisation Guide

## Pipeline Stages and Their Costs

| Stage | Compute | Typical latency | Notes |
|---|---|---|---|
| Regex filter (Step 1) | CPU | < 1 ms | Handles ~70% of standard refusals |
| WildGuard inference (Step 2) | GPU T4 | 50–200 ms | Per-item share when batched |
| Breach formula (Step 3) | CPU | < 0.1 ms | Pure boolean logic |
| MongoDB write | I/O (async) | 2–10 ms | Fire-and-forget, never on critical path |

---

## Batch Size Tuning (`JUDGE_MAX_BATCH`)

WildGuard runs one padded forward pass for an entire batch. CUDA launch overhead is constant, so larger batches give near-linear throughput gains up to memory limits.

| `JUDGE_MAX_BATCH` | Approx. GPU time | VRAM used (extra) | Best for |
|---|---|---|---|
| 1 | 150 ms | baseline | Interactive single-client debugging |
| 8 | 200 ms (÷8 ≈ 25 ms/item) | +400 MB | Low-concurrency server |
| 16 | 300 ms (÷16 ≈ 19 ms/item) | +800 MB | Default — good balance |
| 32 | 500 ms (÷32 ≈ 16 ms/item) | +1.6 GB | High-throughput batch jobs |

Set via environment variable before starting the server:

```bash
export JUDGE_MAX_BATCH=32
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Model Residency (`JUDGE_LAZY_LOAD`)

| Mode | `JUDGE_LAZY_LOAD` | Cold start | Idle VRAM | Best for |
|---|---|---|---|---|
| Resident (default) | `0` | At server startup | ~5.5 GB | Production server |
| Lazy | `1` | First call (~30 s) | 0 GB | Memory-constrained testing |

In resident mode the model is loaded once at `lifespan` startup and shared across all requests. There is no per-request overhead.

---

## Concurrency Model

```
HTTP client A ──┐
HTTP client B ──┤──► asyncio event loop ──► thread executor ──► EnsembleJudge (singleton)
WS client C  ──┘                                                       │
                                                                  GPU (serialised)
```

- The **event loop** handles all I/O concurrently (accepts connections, receives messages, sends responses).
- **GPU inference** is synchronous and serialised. Only one batch runs at a time. Multiple clients queue behind it in the executor's thread pool.
- **MongoDB writes** are dispatched as `asyncio.ensure_future` tasks, so they never delay the response to the client.
- **`MAX_CONCURRENT_WS`** (default 8) limits simultaneous WebSocket sessions to prevent request piling up in the executor queue.

---

## Recommended Deployment Configuration (Tesla T4)

```
VRAM budget  16 GB
──────────────────────────────────────────────────
WildGuard NF4 weights    5.5 GB
KV cache (batch 16)      0.8 GB
Buffer / fragmentation   1.2 GB
──────────────────────────────────────────────────
Free for target model    8.5 GB   ← enough for a second 7B model in NF4
```

```bash
# .env
MONGO_URI=mongodb://user:pass@mongo-host:27017/harmbench
MONGO_PERSIST=1
JUDGE_LAZY_LOAD=0
JUDGE_MAX_BATCH=16
MAX_CONCURRENT_WS=8

# Start
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1 \
        --log-level info --no-access-log
```

Use **nginx** or another reverse proxy in front for TLS termination and to fan multiple clients to the single uvicorn worker:

```nginx
upstream fortiprompt {
    server 127.0.0.1:8000;
}
server {
    listen 443 ssl;
    location / {
        proxy_pass         http://fortiprompt;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 300s;          # allow long evaluation sessions
    }
}
```

---

## Programmatic Batch Evaluation (No Server)

When running offline batch jobs, bypass the server and call the evaluator directly:

```python
from evaluator import MultiTurnEvaluator

sessions = [
    {
        "prompts":   ["seed", "attack payload"],
        "responses": ["OK",   "Here is how to..."],
        "attack_method": "GCG",
        "target_model":  "llama-3-8b",
        "benign_turns":  {0},
    },
    # ... more sessions
]

with MultiTurnEvaluator(mongo_uri="mongodb://localhost:27017", max_turns=10) as ev:
    summaries = ev.run_batch(sessions)

breach_count = sum(1 for s in summaries if s.breached)
print(f"{breach_count}/{len(summaries)} sessions breached")
```

`run_batch()` packs each session into a single `evaluate_batch()` GPU call, so a 10-turn session uses one padded forward pass rather than 10 sequential passes.

---

## Stop-and-Resume

`already_evaluated(session_id, turn_index)` queries the `(session_id, turn_index)` unique index before each turn. Turns already in MongoDB are skipped at zero GPU cost.

To resume an interrupted session, simply re-submit the same request with the same `session_id`:

```bash
# First run (interrupted at turn 4)
curl -X POST http://localhost:8000/evaluate/session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session-abc", "prompts": [...], "responses": [...]}'

# Resume — same session_id, same data
curl -X POST http://localhost:8000/evaluate/session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session-abc", "prompts": [...], "responses": [...]}'
```

---

## Disabling MongoDB (Stateless Mode)

Set `MONGO_PERSIST=0` or pass `"persist": false` in individual requests.

In stateless mode:
- Results are returned directly to the caller (WebSocket / HTTP response).
- No writes to MongoDB.
- Resume / stop-and-resume is unavailable.
- `/report` and `/sessions` endpoints return 503.

Useful for CI smoke tests and environments without a MongoDB instance.

---

## Profiling Individual Turns

Every `TurnResult` carries `latency_ms` and `stage_reached`:

```python
for tr in summary.turn_results:
    if tr.stage_reached == 1:
        print(f"Turn {tr.turn_index}: regex gate ({tr.latency_ms:.1f} ms)")
    else:
        print(f"Turn {tr.turn_index}: GPU inference ({tr.latency_ms:.1f} ms)")
```

A high proportion of `stage_reached == 1` means most responses are caught by the fast regex filter, saving significant GPU compute.
