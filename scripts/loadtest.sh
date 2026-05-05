#!/usr/bin/env bash
# 100-request blocking load test against the queued /predict endpoint.
# Each request enqueues a job, waits for the SSE stream to deliver the
# result, then moves to the next.
set -u
HOST=${HOST:-http://localhost:8000}
N=${N:-100}

texts=(
  "Inflation hit 9.1% in June 2022."
  "I really love the lighting in this room."
  "The 9/11 Commission report said America is safer."
  "What an incredible sunset tonight."
  "Unemployment dropped to 3.5% last quarter."
  "She gave a beautiful speech."
  "The bill was signed into law on March 15."
  "I think this is the best pizza in town."
  "GDP growth slowed to 2.1% year-over-year."
  "Honestly, the new design feels off."
)

start=$(date +%s)
ok=0; fail=0
for i in $(seq 1 "$N"); do
  text="${texts[$((i % ${#texts[@]}))]}"
  job=$(curl -sS -X POST "$HOST/predict" \
        -H 'content-type: application/json' \
        -d "{\"text\": \"$text\"}" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
  # Wait up to 10s for finished/failed event
  result=$(curl -sS --max-time 10 -N "$HOST/predict/$job/stream" 2>/dev/null \
           | awk '/^data:/{print; n++} n>=2{exit}' | tail -1 || true)
  if echo "$result" | grep -q '"is_claim"'; then ok=$((ok+1)); else fail=$((fail+1)); fi
done
end=$(date +%s)
elapsed=$((end - start))
printf '%d ok, %d fail in %ds (%.2f req/s)\n' "$ok" "$fail" "$elapsed" "$(echo "scale=2; $N/$elapsed" | bc -l)"
