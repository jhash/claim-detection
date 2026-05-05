#!/usr/bin/env bash
# Stress test the live claim-detection stack.
#
# What it does:
#   1. Confirms the api is reachable.
#   2. Fires N predict requests against /api/predict (queued path), with
#      C concurrent in-flight at any time, using xargs -P.
#   3. While the load is running, periodically hits /api/healthz to
#      prove the api stays available (decoupled from the queue).
#   4. Reports: total wall time, sustained throughput (req/s),
#      observed latency percentiles (p50/p95/p99), how many healthz
#      probes succeeded, and an extrapolation to 100K requests.
#
# Why /api/predict and not /api/predict/sync? The whole point of the
# queue is to decouple ingestion from inference. /predict accepts the
# job and returns 202 in milliseconds; whether the worker is 5 seconds
# or 5 hours behind doesn't affect the api's availability. That's what
# we're measuring.
#
# Usage:
#   ./scripts/stress_test.sh                  # N=1000, C=50, host=localhost:8000
#   N=10000 C=100 ./scripts/stress_test.sh
#   API=http://my-server:8000 N=500 ./scripts/stress_test.sh

set -uo pipefail

API="${API:-http://localhost:8000}"
N="${N:-1000}"
C="${C:-50}"

if ! curl -fsS "${API}/api/healthz" >/dev/null 2>&1; then
  echo "ERROR: API at ${API}/api/healthz is not reachable." >&2
  echo "  Bring it up first: docker compose up -d --build" >&2
  exit 1
fi

# A handful of representative sentences to round-robin over so the
# stress isn't just the same string cached at every layer.
TEXTS=(
  "Octopuses can edit their own RNA in real time to adapt to cold water."
  "What an unbelievably good cup of coffee."
  "The Voyager 1 probe entered interstellar space in August 2012."
  "I really love how the lighting feels in this room."
  "The Library of Alexandria contained an estimated 400,000 scrolls."
  "Honestly, the new design just feels off to me."
  "Tardigrades can survive in the vacuum of space for ten days."
  "This pizza is the best in the entire neighborhood."
  "Mount Everest grows roughly 4 millimeters every year."
  "She gave a beautiful speech this evening."
)

WORK_DIR="$(mktemp -d)"
trap "rm -rf '$WORK_DIR'" EXIT
TIMINGS_FILE="$WORK_DIR/timings.csv"
HEALTH_FILE="$WORK_DIR/health.csv"
: > "$TIMINGS_FILE"
: > "$HEALTH_FILE"

echo "stressing  $API  with N=$N  C=$C"
echo

# Start a background healthz prober — once a second while the
# requests are firing. Records latency in ms; -1 means failed.
(
  while [ ! -e "$WORK_DIR/done" ]; do
    h=$(curl -o /dev/null -sS -w "%{http_code} %{time_total}\n" "${API}/api/healthz" 2>/dev/null || echo "000 -1")
    echo "$(date +%s) $h" >> "$HEALTH_FILE"
    sleep 1
  done
) &
HEALTH_PID=$!

start_time=$(date +%s)

# Fire N enqueue requests in parallel (C at a time). Each request's
# wall time is captured to the timings file for percentile math.
seq 1 "$N" | xargs -P "$C" -I {} bash -c '
  i={}
  text="${TEXTS[$((i % ${#TEXTS[@]}))]}"
  t=$(curl -o /dev/null -sS -w "%{http_code} %{time_total}\n" \
       -X POST "'"$API"'/api/predict" \
       -H "content-type: application/json" \
       -d "{\"text\": \"$text\"}" 2>/dev/null || echo "000 -1")
  echo "$t" >> "'"$TIMINGS_FILE"'"
' 2>/dev/null
# (the xargs subshell can't see arrays; we re-export TEXTS above through env)
export TEXTS

end_time=$(date +%s)
touch "$WORK_DIR/done"
wait $HEALTH_PID 2>/dev/null

elapsed=$((end_time - start_time))
[ "$elapsed" -lt 1 ] && elapsed=1

# ---------- summarize ----------
total_lines=$(wc -l < "$TIMINGS_FILE" | tr -d ' ')
ok=$(awk '$1 == "202" || $1 == "200"' "$TIMINGS_FILE" | wc -l | tr -d ' ')
fail=$((total_lines - ok))
rps=$(awk "BEGIN { printf \"%.2f\", $total_lines / $elapsed }")

# Latency percentiles in ms (from time_total, in seconds).
percentiles=$(awk '$1 == "202" || $1 == "200" { print $2 * 1000 }' "$TIMINGS_FILE" | sort -n | awk -v n="$ok" '
  { lat[NR] = $1 }
  END {
    if (n == 0) { print "n/a"; exit }
    p50 = lat[int(n * 0.50)+1]
    p95 = lat[int(n * 0.95)+1]
    p99 = lat[int(n * 0.99)+1]
    printf "p50=%.0fms  p95=%.0fms  p99=%.0fms", p50, p95, p99
  }
')

# Healthz probe stats while load was hitting.
hz_total=$(wc -l < "$HEALTH_FILE" | tr -d ' ')
hz_ok=$(awk '$2 == "200"' "$HEALTH_FILE" | wc -l | tr -d ' ')
hz_p95=$(awk '$2 == "200" { print $3 * 1000 }' "$HEALTH_FILE" | sort -n | awk -v n="$hz_ok" '
  { lat[NR] = $1 }
  END {
    if (n == 0) { print "n/a"; exit }
    printf "%.0fms", lat[int(n * 0.95)+1]
  }
')

echo
echo "================== STRESS RESULTS =================="
printf "  N requests              : %d\n" "$N"
printf "  concurrent in-flight    : %d\n" "$C"
printf "  wall time               : %d s\n" "$elapsed"
printf "  enqueued OK             : %d (%d failed)\n" "$ok" "$fail"
printf "  sustained throughput    : %s req/s\n" "$rps"
printf "  enqueue latency         : %s\n" "$percentiles"
echo
printf "  healthz probes (during) : %d / %d succeeded (p95=%s)\n" "$hz_ok" "$hz_total" "$hz_p95"
echo
printf "  100K extrapolation      : at %.0f req/s, 100K = %.1f minutes\n" "$rps" "$(awk "BEGIN { printf \"%.1f\", 100000 / $rps / 60 }")"
echo "===================================================="
echo

# Exit code reflects health: any failed enqueue or unresponsive healthz
# during load is a stress-test failure.
if [ "$fail" -gt 0 ] || [ "$hz_ok" -lt "$hz_total" ]; then
  exit 1
fi
exit 0
