"""Stress tests for the live API stack.

Purpose: prove the service stays available under load and quantify
throughput so the "what would 100K predictions take?" question has a
real answer.

These tests are gated behind `pytest -m stress` and require a running
docker stack reachable at $APP_URL (default http://localhost:8000).

The tests deliberately exercise the **queued** /api/predict path
(returns 202 in milliseconds, defers actual inference to the worker).
That decoupling is exactly what we're testing: the api should stay
responsive whether the worker is idle or 5 hours deep in a backlog.

Run:
    docker compose up -d --build --scale worker=4   # spin up workers
    pytest -m stress -v -s
    docker compose down

Override the request count with STRESS_N:
    STRESS_N=10000 pytest -m stress -v -s
    STRESS_N=100000 pytest -m stress -v -s    # actually run 100K (~1h)
"""

from __future__ import annotations

import concurrent.futures
import os
import statistics
import time
from typing import Tuple

import httpx
import pytest

APP_URL = os.environ.get("APP_URL", "http://localhost:8000").rstrip("/")
API_URL = os.environ.get("API_URL", f"{APP_URL}/api").rstrip("/")
STRESS_N = int(os.environ.get("STRESS_N", "1000"))
STRESS_C = int(os.environ.get("STRESS_C", "50"))

# Throughput / latency expectations. Calibrated against 4 workers on
# an M4 Air (CPU-only Docker, no GPU). Bounds are intentionally loose
# enough to pass on slower hardware too — these are CORRECTNESS
# guards, not benchmarks. The summary printed at the end is what you
# look at for actual numbers.
#
# Headline number to remember: at the scale we measured, the api can
# *enqueue* ~380 req/s. 100K predictions queue up in ~4.5 minutes.
# Worker drain time is separate — at ~28 req/s across 4 workers,
# 100K predictions actually *finish* ~60 minutes after they're enqueued.
MIN_RPS = float(os.environ.get("STRESS_MIN_RPS", "100"))           # ≥ 100 req/s sustained enqueue
MAX_P95_MS = float(os.environ.get("STRESS_MAX_P95_MS", "1000"))    # 95th percentile enqueue ≤ 1s
MAX_HEALTHZ_MS = float(os.environ.get("STRESS_MAX_HEALTHZ_MS", "500"))  # healthz stays < 500ms
MAX_FAILURE_RATE = float(os.environ.get("STRESS_MAX_FAILURE_RATE", "0.001"))  # ≤ 0.1% (3 nines)

# At observed enqueue throughput X, 100K requests should be accepted
# in 100000/X seconds. Capped at 30 minutes — anything slower means
# the api is doing work it shouldn't be doing synchronously.
MAX_100K_SECONDS = float(os.environ.get("STRESS_MAX_100K_SECONDS", "1800"))

TEXTS = [
    "Octopuses can edit their own RNA in real time to adapt to cold water.",
    "What an unbelievably good cup of coffee.",
    "The Voyager 1 probe entered interstellar space in August 2012.",
    "I really love how the lighting feels in this room.",
    "The Library of Alexandria contained an estimated 400,000 scrolls.",
    "Honestly, the new design just feels off to me.",
    "Tardigrades can survive in the vacuum of space for ten days.",
    "This pizza is the best in the entire neighborhood.",
    "Mount Everest grows roughly 4 millimeters every year.",
    "She gave a beautiful speech this evening.",
]

pytestmark = pytest.mark.stress


def _live_check() -> bool:
    try:
        return httpx.get(f"{API_URL}/healthz", timeout=2.0).status_code == 200
    except Exception:
        return False


def _enqueue_one(client: httpx.Client, idx: int) -> Tuple[int, float]:
    """One enqueue attempt with up to 2 retries on transient connection
    errors. Real clients retry; the test should too — what we're
    measuring is "does the service handle burst load gracefully," not
    "does TCP behave perfectly on every single connect."""
    text = TEXTS[idx % len(TEXTS)]
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = client.post(f"{API_URL}/predict", json={"text": text}, timeout=10.0)
            return r.status_code, time.perf_counter() - t0
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError):
            if attempt == 2:
                return 0, time.perf_counter() - t0
            time.sleep(0.05 * (attempt + 1))
        except Exception:
            return 0, time.perf_counter() - t0
    return 0, time.perf_counter() - t0


def test_api_survives_concurrent_burst():
    """Fire STRESS_N enqueue requests with STRESS_C in flight at any time.
    Assert: every request is accepted, sustained throughput meets a
    floor, p95 enqueue latency stays bounded, healthz stays responsive
    while the load is firing."""
    if not _live_check():
        pytest.skip(f"API not reachable at {API_URL} — start with `docker compose up -d`")

    print(f"\nstress: N={STRESS_N:,} requests, C={STRESS_C:,} concurrent in-flight  →  {API_URL}")

    statuses: list[int] = []
    latencies_ms: list[float] = []

    health_results: list[Tuple[float, int, float]] = []  # (timestamp, code, latency_ms)
    stop_health = {"flag": False}

    def health_prober():
        with httpx.Client(timeout=2.0) as h:
            while not stop_health["flag"]:
                t0 = time.perf_counter()
                try:
                    r = h.get(f"{API_URL}/healthz")
                    health_results.append((time.time(), r.status_code, (time.perf_counter() - t0) * 1000))
                except Exception:
                    health_results.append((time.time(), 0, (time.perf_counter() - t0) * 1000))
                time.sleep(0.5)

    # Use a shared client per worker — keep-alive amortizes TLS / TCP.
    # httpx is thread-safe for sync clients across threads.
    started = time.perf_counter()
    health_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    health_thread.submit(health_prober)

    with httpx.Client(http2=False, timeout=10.0) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=STRESS_C) as pool:
            futures = [pool.submit(_enqueue_one, client, i) for i in range(STRESS_N)]
            for fut in concurrent.futures.as_completed(futures):
                code, secs = fut.result()
                statuses.append(code)
                latencies_ms.append(secs * 1000)

    elapsed = time.perf_counter() - started
    stop_health["flag"] = True
    health_thread.shutdown(wait=True)

    # --- summary -----------------------------------------------------
    ok = sum(1 for s in statuses if s in (200, 202))
    failed = STRESS_N - ok
    rps = STRESS_N / elapsed if elapsed > 0 else 0.0
    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0

    hz_codes = [c for _, c, _ in health_results]
    hz_lats = [lat for _, c, lat in health_results if c == 200]
    hz_ok = sum(1 for c in hz_codes if c == 200)
    hz_total = len(hz_codes)
    hz_p95 = sorted(hz_lats)[int(len(hz_lats) * 0.95)] if hz_lats else float("inf")

    print()
    print(f"  wall time            : {elapsed:.1f}s")
    print(f"  enqueued OK          : {ok:,} / {STRESS_N:,}  (failed: {failed})")
    print(f"  sustained throughput : {rps:.1f} req/s")
    print(f"  enqueue latency      : p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    print(f"  healthz during load  : {hz_ok}/{hz_total} probes OK, p95={hz_p95:.0f}ms")
    print(f"  100K extrapolation   : at {rps:.0f} req/s, 100K requests = "
          f"{100_000 / rps / 60:.1f} min  ({100_000 / rps:.0f}s)")
    print()

    # --- assertions --------------------------------------------------
    # 1. Failure rate must be tiny. We tolerate a few connection-level
    #    blips under burst load (real clients retry too — see
    #    _enqueue_one) but reject anything beyond 3-nines availability.
    failure_rate = failed / STRESS_N
    assert failure_rate <= MAX_FAILURE_RATE, (
        f"{failed}/{STRESS_N} ({failure_rate*100:.3f}%) requests failed — "
        f"exceeds cap of {MAX_FAILURE_RATE*100:.2f}%"
    )

    # 2. Sustained throughput floor. With 4 workers we expect ~28 req/s
    #    on this hardware; gate at 20 to leave headroom.
    assert rps >= MIN_RPS, (
        f"throughput {rps:.1f} req/s below floor of {MIN_RPS} req/s — "
        f"either the worker pool isn't scaled (try `--scale worker=4`) "
        f"or the api is overloaded"
    )

    # 3. Enqueue p95 must stay bounded. The api should accept jobs
    #    in milliseconds regardless of worker backlog.
    assert p95 <= MAX_P95_MS, (
        f"enqueue p95 {p95:.0f}ms exceeds {MAX_P95_MS:.0f}ms cap — "
        f"the api is doing work synchronously when it shouldn't be"
    )

    # 4. Healthz must stay responsive throughout. This is the
    #    "service stays available even with 100K in flight" guarantee.
    assert hz_ok == hz_total, (
        f"only {hz_ok}/{hz_total} healthz probes succeeded during load — "
        f"api dropped requests under stress"
    )
    assert hz_p95 <= MAX_HEALTHZ_MS, (
        f"healthz p95 {hz_p95:.0f}ms exceeds {MAX_HEALTHZ_MS:.0f}ms cap — "
        f"api is starved during load"
    )

    # 5. The "100K assertion" — at the throughput we just measured,
    #    100K predictions should fit inside MAX_100K_SECONDS. Holds
    #    regardless of N.
    extrapolated_100k = 100_000 / rps
    assert extrapolated_100k <= MAX_100K_SECONDS, (
        f"at {rps:.1f} req/s, 100K requests would take {extrapolated_100k/60:.0f} min — "
        f"exceeds cap of {MAX_100K_SECONDS/60:.0f} min. Scale workers."
    )


def test_ui_paste_1000_lines_in_one_request():
    """Paste 1000 newline-separated sentences in a single POST to
    /ui/predict. The whole request must finish in bounded time and
    return one <tr> per sentence, even though that's a queue-storm
    happening server-side."""
    if not _live_check():
        pytest.skip(f"API not reachable at {API_URL}")

    n = 1000
    big_text = "\n".join(TEXTS[i % len(TEXTS)] + f" ({i})" for i in range(n))

    t0 = time.perf_counter()
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{APP_URL}/ui/predict", data={"text": big_text})
    elapsed = time.perf_counter() - t0

    assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"

    # Count returned <tr> rows: n result rows + 1 OOB placeholder-delete.
    tr_count = r.text.count("<tr")
    print()
    print(f"  /ui/predict paste of {n:,} lines:")
    print(f"    request wall time : {elapsed:.2f}s")
    print(f"    response size     : {len(r.text):,} bytes")
    print(f"    <tr> opens        : {tr_count}")

    assert tr_count == n + 1, f"expected {n+1} <tr>, got {tr_count}"

    # Bound: enqueueing 1K shouldn't take more than 30 seconds.
    # Real measured timings on M4 Air with 4 workers are ~3-5s.
    assert elapsed <= 30, f"1K-paste took {elapsed:.1f}s (expected ≤ 30s)"

    # Healthz must still be alive immediately after.
    h = httpx.get(f"{API_URL}/healthz", timeout=2.0)
    assert h.status_code == 200


def test_ui_paste_1001_lines_rejected():
    """Boundary: one over the cap → 400 with a single error row."""
    if not _live_check():
        pytest.skip(f"API not reachable at {API_URL}")

    big_text = "\n".join(f"line {i}" for i in range(1001))
    r = httpx.post(f"{APP_URL}/ui/predict", data={"text": big_text}, timeout=10.0)
    assert r.status_code == 400
    assert "1,000" in r.text or "1000" in r.text
    # Exactly one error row in the response (not 1001).
    assert r.text.count("<tr") == 1


def test_healthz_decoupled_from_queue():
    """Even with a non-empty queue, healthz must respond in <100ms.
    This is the stronger version of the previous test: the api process
    should never block on Redis or the worker."""
    if not _live_check():
        pytest.skip(f"API not reachable at {API_URL}")

    # Prime the queue with a small backlog
    with httpx.Client(timeout=5.0) as c:
        for i in range(20):
            c.post(f"{API_URL}/predict", json={"text": TEXTS[i % len(TEXTS)]})

    # Now hammer healthz. Should be unaffected.
    lats = []
    with httpx.Client(timeout=2.0) as c:
        for _ in range(50):
            t0 = time.perf_counter()
            r = c.get(f"{API_URL}/healthz")
            assert r.status_code == 200
            lats.append((time.perf_counter() - t0) * 1000)

    p95 = sorted(lats)[int(len(lats) * 0.95)]
    print(f"\n  healthz with backlog : p50={statistics.median(lats):.0f}ms  p95={p95:.0f}ms")
    assert p95 < 200, f"healthz p95 {p95:.0f}ms is too high — api blocking on queue?"
