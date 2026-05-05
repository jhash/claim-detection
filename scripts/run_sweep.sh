#!/usr/bin/env bash
# Sequential train + eval + compare sweep over the model registry.
#
# - Trains each model in registry order (src/models.py).
# - Skips training if runs/<slug>/final/ already exists (resumable).
# - Skips eval if results/<slug>.json already exists.
# - Re-runs compare.py after every successful eval so RESULTS.md is
#   always current to the most recently finished model.
#
# Usage:
#   ./scripts/run_sweep.sh                  # sweep all
#   ./scripts/run_sweep.sh ettin-150m-ft    # one model only

set -u  # don't -e: a single failed model shouldn't kill the whole sweep
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=runs/_logs
mkdir -p "$LOG_DIR"

if [ $# -gt 0 ]; then
  SLUGS=("$@")
else
  mapfile -t SLUGS < <("$PY" -m src.pipeline --list 2>/dev/null | awk '{print $1}' | grep -v '^└─' | grep -v '^$')
fi

echo "sweep over ${#SLUGS[@]} model(s):"
printf '  - %s\n' "${SLUGS[@]}"
echo

for slug in "${SLUGS[@]}"; do
  echo "=========================================================="
  echo "[$(date +%H:%M:%S)]  $slug"
  echo "=========================================================="

  if [ -d "runs/$slug/final" ]; then
    echo "  train: skip (runs/$slug/final/ exists)"
  else
    echo "  train: starting..."
    "$PY" -m src.pipeline --model "$slug" 2>&1 | tee "$LOG_DIR/${slug}.train.log" \
      | grep --line-buffered -E '^\{|epoch|FAILED|error|ERROR|elapsed_sec' || true
  fi

  if [ -f "results/$slug.json" ]; then
    echo "  eval:  skip (results/$slug.json exists)"
  else
    echo "  eval:  starting..."
    "$PY" -m src.evaluate --model "$slug" 2>&1 | tee "$LOG_DIR/${slug}.eval.log" \
      | grep --line-buffered -E '"f1"|"accuracy"|FAILED|error' || true
  fi

  echo "  compare: regenerating RESULTS.md..."
  "$PY" -m src.compare
  echo
done

echo "sweep done."
