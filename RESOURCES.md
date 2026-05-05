# Resource usage

Captured live from the Docker stack on a MacBook Air M4 (32 GB unified
memory, 10 cores). Docker Desktop default container resource limits
apply (~7.75 GB RAM cap per container, ~2 logical CPUs).

## Idle baseline (containers up, no traffic)

| Container | CPU | RAM | Notes |
|---|---:|---:|---|
| `api`     | 0.2% | **298 MB** | FastAPI + uvicorn, no model loaded |
| `worker`  | 0.0% | **794 MB** | RQ SimpleWorker + Ettin-150m resident in memory |
| `redis`   | 0.4% | **15 MB**  | empty queue + result store |

The worker is heavier than the API on purpose: the API is just async glue
(receive → enqueue → poll Redis), and the worker holds the model resident
across jobs.

## Under load (queued mode)

| Container | CPU peak | RAM | Notes |
|---|---:|---:|---|
| `api`     | 2.4%    | 298 MB | mostly idle — async I/O + 50 ms Redis poll per active stream |
| `worker`  | **88%** | 794 MB | one core busy when serving sequentially; OpenMP-parallelism would push this to 180+% with batched workloads |
| `redis`   | 1.0%    | 15 MB  | invisible |

Worker memory is steady at ~800 MB — the model stays resident, no
per-job re-load. (Earlier numbers showed ~470 MB bobbing because the
default `rq worker` *forked per job*, throwing the cache out every time
— see "Latency journey" below.)

## Per-request latency (queued mode, final)

End-to-end **~150 ms steady state**, including queueing + inference +
SSE round-trip.

| Phase | Wall clock |
|---|---:|
| Cold start (first request after worker boot) | 3–5 s |
| Steady state (warm worker, queued + SSE) | **~150 ms** |
| Steady state (sync mode, no queue) | ~50–80 ms |

### Latency journey (each fix)

| Iteration | Round-trip | What was wrong |
|---|---:|---|
| Original — `asyncio.sleep(0.5)` poll | **~2,000 ms** | Floor was the polling interval, not inference |
| Pub/sub via redis-py | ~7,000 ms | Synchronous get_message blocked the asyncio event loop |
| Pub/sub via to_thread | ~10,000 ms (timeout) | redis-py SUBSCRIBE ack races with our publish — events lost |
| Pub/sub + SimpleWorker | mixed (130 ms / timeout) | Latent fork-per-job forced model reload (1.9 s/job) |
| **50 ms async polling** | **~150 ms** ✅ | Right size of hammer for the problem |

**Root cause** of the original 2 s wasn't actually polling cadence — it
was that **RQ's default worker forks per job**, which throws away the
in-process model cache. Fixed by switching to `rq.SimpleWorker` in
`docker-compose.yml`. After that, even a 50 ms poll is fast enough.

## Storage

### Trained checkpoints (your local `runs/`)

```
runs/                                                 12 GB total
├── ettin-150m-ft/            2.3 GB  (final: 574 MB)  ⭐ headline
├── ettin-150m-pretrained/    1.1 GB  (final: 574 MB)
├── modernbert-base-ft/       2.3 GB  (final: 574 MB)
├── modernbert-base-pretrained/ 1.1 GB (final: 574 MB)
├── bert-base-ft/             1.6 GB  (final: 418 MB)
├── bert-base-pretrained/     837 MB  (final: 418 MB)
├── roberta-base-ft/          1.9 GB  (final: 479 MB)
└── roberta-base-pretrained/  962 MB  (final: 479 MB)
```

Bulk is the `ckpt/` subdirectories — intermediate per-epoch checkpoints
HF Trainer writes during training. Only `runs/<slug>/final/` is needed
for inference. To free ~10 GB:

```bash
rm -rf runs/*/ckpt
```

### Raw pretrained models (HuggingFace cache)

```
~/.cache/huggingface/                                 2.6 GB total
├── google-bert/bert-base-uncased       421 MB
├── FacebookAI/roberta-base             478 MB
├── answerdotai/ModernBERT-base         573 MB
└── jhu-clsp/ettin-encoder-150m        1.1 GB
```

Auto-downloaded at first use of each model. Safe to delete (re-downloads
on next training run). Relocate via `HF_HOME=/path/to/somewhere`.

### Just the Ettin checkpoint (the only thing needed in production)

```
runs/ettin-150m-ft/final/    574 MB
├── model.safetensors        571 MB    ← actual weights
├── tokenizer.json           3.4 MB
├── config.json              2 KB
├── tokenizer_config.json    1 KB
└── training_args.bin        5 KB      ← can delete, not needed for inference
```

## Expected on other hardware

| Hardware | Per-request latency (warm) | Notes |
|---|---:|---|
| **M4 Air** (this machine, MPS) | ~80 ms | Docker on Mac runs containers on a Linux VM; no MPS inside the container, so this is CPU |
| **MacBook Pro M1 Max/Ultra** | ~70 ms | M1 series is one generation behind M4 but still strong |
| **Modern x86 server** (16+ cores, AVX2, no GPU) | ~80–120 ms | What `DEPLOY_LINUX.md` is targeted at |
| **Older x86** (4–8 cores, no AVX2) | ~250–400 ms | Still works, just slower |
| **Bare CPU laptop** (single core saturation) | ~600 ms+ | Set `OMP_NUM_THREADS=1` to avoid thrash |

The model is small (574 MB on disk, ~2 GB resident with PyTorch overhead),
so RAM isn't the bottleneck on any reasonable hardware. The workload is
matmul-bound, so SIMD width and core count drive latency.

## Scaling guidance

For a single API server handling more than ~10 req/s, scale workers
horizontally:

```bash
# In Docker Compose
docker compose up -d --scale worker=4

# Or in Swarm
docker service scale claim-detection_worker=4
```

Each worker holds its own copy of the model (~2 GB resident), so plan
for `2 GB × N workers` plus a bit of headroom for the API container
itself (~300 MB).

Set `OMP_NUM_THREADS` per-worker to avoid contention when multiple
workers share a host:

```yaml
environment:
  OMP_NUM_THREADS: "2"   # if you have 8 cores and 4 workers
```

## How to capture this yourself

```bash
# One-shot snapshot
docker stats --no-stream

# Continuous (refresh every 1s, ctrl-C to stop)
docker stats

# 100-request load test
N=100 ./scripts/loadtest.sh
```
