# Deploying the Ettin-150m fine-tuned model to a CPU-only Linux server

This guide moves the trained model + the API stack from this Mac to a
Linux server that has no GPU. Two paths — pick the one that fits your
operational style.

## What you're moving

The fine-tuned checkpoint lives at:

```
runs/ettin-150m-ft/final/
├── config.json              2 KB    — architecture description
├── model.safetensors      571 MB    — the actual model weights
├── tokenizer.json         3.4 MB    — token → id mapping
├── tokenizer_config.json    1 KB    — tokenizer settings
└── training_args.bin       5 KB     — HF Trainer args (not needed for inference)
```

**Total: ~574 MB.** Everything else under `runs/` is intermediate
checkpoint state from training — you don't need it on the server.

You can drop `training_args.bin` if you want; nothing at inference time
reads it.

---

## Path A — Docker Swarm (recommended if you already use Swarm)

The `docker-compose.yml` in this repo already deploys cleanly to Swarm.
Two extra steps because the model needs to be *on the server* for the
container to mount it:

### 1. Copy the checkpoint to the server

```bash
# from this Mac
rsync -avz --progress runs/ettin-150m-ft/final/ \
    user@server:/srv/claim-detection/model/
```

`rsync` is ~574 MB once; future redeploys with the same model are
instant (rsync diffs).

### 2. Copy the repo (without the heavy stuff)

```bash
rsync -avz --progress \
    --exclude='runs' --exclude='.venv' --exclude='datasets' \
    --exclude='papers' --exclude='__pycache__' --exclude='.git' \
    /Users/jakehash/Development/jobs/claim-detection/ \
    user@server:/srv/claim-detection/repo/
```

(Or `git clone` on the server — the repo without `runs/`, `datasets/`,
and `papers/` is small.)

### 3. Adjust the compose volume mount

On the server, edit `docker-compose.yml` so both `api` and `worker` mount
the model from where you copied it:

```yaml
volumes:
  - /srv/claim-detection/model:/model:ro
```

(replacing `./runs/ettin-150m-ft/final`).

### 4. Build the image and deploy

```bash
ssh user@server
cd /srv/claim-detection/repo
docker build -t claim-detection:latest -f docker/Dockerfile .
docker stack deploy -c docker-compose.yml claim-detection
docker service ls
```

That's it. The Dockerfile already pins **CPU-only torch**
(`torch==2.6.0+cpu`), so no CUDA needed and no GPU detection drama.

### Scale workers if needed

```bash
docker service scale claim-detection_worker=4
```

### Health-check the deployment

```bash
curl http://server:8000/healthz
# {"status":"ok","queue":true}

curl -X POST http://server:8000/predict \
    -H 'content-type: application/json' \
    -d '{"text": "Inflation hit 9.1% in June 2022."}'
```

---

## Path B — bake the model into the image

If you don't want a separate volume mount (e.g. you're deploying to a
constrained PaaS, or the swarm has nodes without persistent storage),
copy the checkpoint *into* the image at build time:

### 1. Modify the Dockerfile

Add this near the bottom of `docker/Dockerfile`, before `USER app`:

```dockerfile
COPY runs/ettin-150m-ft/final/ /model/
```

### 2. Modify `.dockerignore`

The current `.dockerignore` excludes `runs/`. Either remove that line
or scope it tighter:

```
runs/*
!runs/ettin-150m-ft/
!runs/ettin-150m-ft/final/
```

### 3. Build and push to a registry

```bash
docker build -t your-registry/claim-detection:0.1.0 -f docker/Dockerfile .
docker push your-registry/claim-detection:0.1.0
```

The image will be ~1.4 GB instead of ~800 MB — bigger but completely
self-contained. The volume-mount config in `docker-compose.yml` becomes
redundant; you can drop it.

### 4. Deploy

```bash
docker service create --name claim-api --publish 8000:8000 \
    your-registry/claim-detection:0.1.0
```

---

## Path C — bare-metal (no Docker, just systemd)

If your server doesn't run Docker:

```bash
ssh user@server

# Python deps
sudo apt install -y python3.13 python3.13-venv
python3.13 -m venv /opt/claim-detection/.venv
source /opt/claim-detection/.venv/bin/activate

# CPU-only torch (much smaller than the default wheel)
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
pip install fastapi 'uvicorn[standard]' jinja2 transformers safetensors python-multipart

# Copy code + model (from your Mac)
# rsync the repo (minus runs/) and runs/ettin-150m-ft/final/ to /opt/claim-detection/
```

`/etc/systemd/system/claim-detection.service`:

```ini
[Unit]
Description=Claim Detection API
After=network.target

[Service]
User=app
WorkingDirectory=/opt/claim-detection/repo
Environment="MODEL_PATH=/opt/claim-detection/model"
Environment="QUEUE=0"
ExecStart=/opt/claim-detection/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now claim-detection
sudo systemctl status claim-detection
curl http://localhost:8000/healthz
```

(If you want the queued mode, you also need Redis + an `rq worker`
service unit. Easier to just run sync mode here and use Docker if you
need the queue.)

---

## What to expect on a CPU-only server

This was trained on Apple Silicon's MPS GPU (~12 samples/sec). Inference
on a server CPU will be slower per request:

| Hardware | Predicted latency for one sentence |
|---|---|
| M4 Air (MPS, this machine) | ~15 ms |
| Modern x86-64 server (16+ cores, AVX2) | ~80 ms |
| Older x86-64 (4–8 cores, no AVX2) | ~300 ms |

For batched inference (multiple sentences at once), throughput scales
roughly linearly until you saturate cache. Practical ballpark: **a
modest 8-core server should handle ~50–100 sentences/sec** in queued
mode with one worker; scale workers horizontally if you need more.

CPU torch uses MKL (Intel) or OpenBLAS (AMD/ARM) under the hood, so make
sure those libraries are installed (`apt install libopenblas0-pthread`).

## Tips and pitfalls

- **Don't ship the venv** — Python venvs aren't relocatable across
  hosts. Always pip-install fresh on the target.
- **Don't ship `~/.cache/huggingface`** — the Dockerfile sets
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the container
  will never reach out to HuggingFace at runtime. Everything it needs
  is in `/model`.
- **Memory**: peak resident set for one worker loading Ettin is ~2 GB.
  Plan for `2 GB × N workers` plus a bit of headroom for the API itself.
- **Cold start**: first request after a worker spins up will be slow
  (~3–5 s) as PyTorch initializes and the safetensors file gets paged
  in. Subsequent requests are warm.
- **Threads**: torch defaults to spawning one inference thread per
  physical CPU core. If you run multiple workers on the same host,
  set `OMP_NUM_THREADS=2` (or similar) so they don't fight for cores.
