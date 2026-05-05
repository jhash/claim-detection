# Running the API in Docker

## Local: `docker compose up`

Prereqs: Docker Desktop running, the Ettin checkpoint at
`runs/ettin-150m-ft/final/` (run `python -m src.pipeline --model
ettin-150m-ft` if it doesn't exist).

```bash
docker compose up --build              # foreground, three services
# or
docker compose up -d --build           # background
```

This brings up three containers:

| service | role |
|---|---|
| `api`     | FastAPI on `http://localhost:8000` — serves the UI, accepts predict requests, enqueues jobs |
| `worker`  | RQ worker that loads the model once and processes jobs from Redis |
| `redis`   | job queue + result store |

Both `api` and `worker` mount `runs/ettin-150m-ft/final/` read-only at
`/model` inside the container, so changing the model is just a
recompose with a different volume.

### Verify it's reachable

```bash
curl http://localhost:8000/healthz
# {"status":"ok","queue":true}

# Queued predict — returns a job id, then SSE-stream the result.
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Inflation hit 9.1% in June 2022."}'
# {"job_id":"...","stream_url":"/predict/.../stream"}

curl -N http://localhost:8000/predict/<job_id>/stream
# event: status\ndata: {"status":"queued"}
# event: status\ndata: {"status":"started"}
# event: result\ndata: {"is_claim":true,"confidence":0.96,"label":"claim"}
```

UI at <http://localhost:8000/>, comparison table at
<http://localhost:8000/results>.

### Run integration tests against the live stack

The `pytest -m integration` set is excluded by default (`pyproject.toml`).
With the docker stack up:

```bash
.venv/bin/pytest -m integration -v
```

These hit the real container, exercise the full HTTP path including the
queue, and intentionally fail when the stack isn't running — that's the
"failing-first integration test" called out in the takehome.

## Server: Docker Swarm deploy

The same `docker-compose.yml` file deploys cleanly to Swarm (it uses
v3 syntax, no `depends_on.condition`, declared `deploy.replicas`):

```bash
# On the swarm manager:
docker stack deploy -c docker-compose.yml claim-detection
docker service ls
```

Scale the worker independently of the API:

```bash
docker service scale claim-detection_worker=4
```

Note: the Swarm deploy expects the model directory to be available on
each node where `worker` or `api` lands. Either:

- Bake the model into a custom image (build with
  `--build-arg MODEL_DIR=...` and `COPY` it in), or
- Use a shared volume / NFS mount across nodes, or
- Constrain placement so workers only run on nodes that have the model
  directory locally:

  ```yaml
  deploy:
    placement:
      constraints: [node.labels.has_model == true]
  ```

  Then label the node: `docker node update --label-add has_model=true <node>`.

## Container architecture

- **One image, two roles**: same `claim-detection:latest` image runs as
  both the API (`uvicorn …`) and the worker (`rq worker …`). The role is
  set by the `command:` field. Halves what we maintain.
- **CPU-only torch**: the Dockerfile installs `torch==2.6.0+cpu` from
  PyTorch's CPU-only index. Image stays around 800 MB instead of 4+ GB
  with CUDA. MPS doesn't work in containers anyway.
- **Offline transformers**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  are set in the runtime image so a cold-start container won't try to
  reach the HuggingFace Hub. Everything it needs is in `/model`.
- **Healthcheck**: every 30 s `curl -fsS /healthz`. Compose / Swarm
  restart on failure.
