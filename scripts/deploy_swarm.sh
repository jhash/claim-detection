#!/usr/bin/env bash
# Blue-green-style rolling deploy of the claim-detection stack to a
# local single-node (or multi-node) Docker Swarm.
#
# What this does, in order:
#   1. Pulls latest source on this host (`git pull`).
#   2. Computes an immutable image tag from the current git short SHA.
#   3. Builds the new image at that tag (does NOT touch running services yet).
#   4. Smoke-tests the image stand-alone — `curl /api/healthz` against a
#      one-shot container. Aborts the deploy if smoke fails.
#   5. `docker stack deploy`s the new tag. Swarm rolls tasks one at a
#      time with `order: start-first` — a NEW task starts on the new
#      image, becomes healthy, THEN the old task shuts down. Zero
#      downtime as long as `API_REPLICAS >= 2`.
#   6. Watches the rollout, waits until both api and worker services
#      converge to the new image (or auto-rollback fires).
#   7. Hits the live `/api/healthz` and prints the running tag, so a
#      human can verify before walking away.
#
# Usage from the project root on the swarm host:
#   ./scripts/deploy_swarm.sh                  # build + deploy
#   ./scripts/deploy_swarm.sh --tag abc1234    # deploy a specific tag (no build)
#   ./scripts/deploy_swarm.sh --skip-pull      # skip `git pull`
#   ./scripts/deploy_swarm.sh --skip-smoke     # skip stand-alone container smoke

set -euo pipefail

# ---------- defaults ----------
STACK="${STACK:-claim-detection}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
API_REPLICAS="${API_REPLICAS:-2}"     # 2 → zero-downtime rolls
WORKER_REPLICAS="${WORKER_REPLICAS:-1}"
SMOKE_PORT="${SMOKE_PORT:-18000}"
SMOKE_NAME="claim-detection-smoke"
HOST_PORT="${HOST_PORT:-8000}"

DO_PULL=1
DO_BUILD=1
DO_SMOKE=1
EXPLICIT_TAG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         EXPLICIT_TAG="$2"; DO_BUILD=0; shift 2 ;;
    --skip-pull)   DO_PULL=0; shift ;;
    --skip-build)  DO_BUILD=0; shift ;;
    --skip-smoke)  DO_SMOKE=0; shift ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---------- pre-flight ----------
if ! docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -qx active; then
  echo "ERROR: this node is not a Swarm manager." >&2
  echo "  Run: docker swarm init" >&2
  exit 1
fi

if [ "$DO_PULL" = 1 ]; then
  echo "==> git pull"
  git pull --ff-only origin main
fi

# ---------- choose tag ----------
if [ -n "$EXPLICIT_TAG" ]; then
  TAG="$EXPLICIT_TAG"
else
  TAG="$(git rev-parse --short HEAD)"
fi
IMAGE="claim-detection:$TAG"
echo "==> deploying $IMAGE  (stack=$STACK, api_replicas=$API_REPLICAS)"

# ---------- build ----------
if [ "$DO_BUILD" = 1 ]; then
  echo "==> docker build → $IMAGE"
  docker build -t "$IMAGE" -f docker/Dockerfile .
else
  echo "==> skipping build (using existing $IMAGE)"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image $IMAGE not present locally and build skipped." >&2
    exit 1
  fi
fi

# ---------- stand-alone smoke ----------
if [ "$DO_SMOKE" = 1 ]; then
  echo "==> smoke test on host port $SMOKE_PORT"
  docker rm -f "$SMOKE_NAME" >/dev/null 2>&1 || true
  docker run --rm -d --name "$SMOKE_NAME" \
    -p "$SMOKE_PORT:8000" \
    -v "$ROOT/runs/ettin-150m-ft/final:/model:ro" \
    -v "$ROOT/results:/app/results:ro" \
    -e QUEUE=0 \
    "$IMAGE" >/dev/null

  cleanup_smoke() { docker stop "$SMOKE_NAME" >/dev/null 2>&1 || true; }
  trap cleanup_smoke EXIT

  printf "    waiting for healthz "
  for i in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$SMOKE_PORT/api/healthz" >/dev/null 2>&1; then
      echo " OK"
      break
    fi
    printf "."
    sleep 1
    if [ "$i" = 20 ]; then
      echo " FAILED"
      echo "==> smoke logs:"
      docker logs "$SMOKE_NAME" 2>&1 | tail -40
      cleanup_smoke
      trap - EXIT
      exit 1
    fi
  done
  cleanup_smoke
  trap - EXIT
fi

# ---------- deploy to swarm ----------
echo "==> docker stack deploy"
TAG="$TAG" API_REPLICAS="$API_REPLICAS" WORKER_REPLICAS="$WORKER_REPLICAS" \
  docker stack deploy -c "$COMPOSE_FILE" --with-registry-auth "$STACK"

# ---------- watch the roll ----------
echo "==> waiting for services to converge to $IMAGE (or auto-rollback)"
deadline=$(( $(date +%s) + 300 ))   # 5 minutes hard cap
while :; do
  api_image=$(docker service inspect --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}'     "${STACK}_api" 2>/dev/null | sed 's/@.*//')
  worker_image=$(docker service inspect --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}'     "${STACK}_worker" 2>/dev/null | sed 's/@.*//')

  api_state=$(docker service inspect --format '{{.UpdateStatus.State}}' "${STACK}_api" 2>/dev/null || echo "")
  worker_state=$(docker service inspect --format '{{.UpdateStatus.State}}' "${STACK}_worker" 2>/dev/null || echo "")

  api_replicas=$(docker service ls --filter "name=${STACK}_api" --format '{{.Replicas}}' | head -n1)
  worker_replicas=$(docker service ls --filter "name=${STACK}_worker" --format '{{.Replicas}}' | head -n1)

  api_ready=0
  worker_ready=0

  if [ "$api_image" = "$IMAGE" ] && [[ "$api_replicas" =~ ^([0-9]+)/\1$ ]] && { [ "$api_state" = "completed" ] || [ -z "$api_state" ]; }; then
    api_ready=1
  fi
  if [ "$worker_image" = "$IMAGE" ] && [[ "$worker_replicas" =~ ^([0-9]+)/\1$ ]] && { [ "$worker_state" = "completed" ] || [ -z "$worker_state" ]; }; then
    worker_ready=1
  fi

  printf "    api=%s (%s,%s)  worker=%s (%s,%s)\n"     "${api_state:-none}" "$api_image" "${api_replicas:-?}"     "${worker_state:-none}" "$worker_image" "${worker_replicas:-?}"

  if [ "$api_ready" = "1" ] && [ "$worker_ready" = "1" ]; then
    echo "==> rolled successfully"
    break
  fi
  if [ "$api_state" = "rollback_completed" ] || [ "$worker_state" = "rollback_completed" ]; then
    echo "ERROR: deploy auto-rolled back. New tasks failed health checks." >&2
    docker service ps "${STACK}_api" --no-trunc | head
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "ERROR: roll did not converge within 5 minutes." >&2
    docker service ps "${STACK}_api" --no-trunc | head
    exit 1
  fi
  sleep 3
done

# ---------- post-deploy verify ----------
echo
echo "==> live verification"
docker service ls --filter "name=${STACK}"
echo
echo "running images:"
echo "  api:    $(docker service inspect --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' "${STACK}_api" | sed 's/@.*//')"
echo "  worker: $(docker service inspect --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' "${STACK}_worker" | sed 's/@.*//')"
echo
if curl -fsS "http://127.0.0.1:${HOST_PORT}/api/healthz" 2>/dev/null; then
  echo
  echo "==> deployed $IMAGE — healthz OK"
else
  echo "warning: /api/healthz on host port $HOST_PORT didn't respond." >&2
  echo "  (the swarm may have published it to a different port — check with \`docker service ls\`)" >&2
fi
