"""Redis-backed background queue for predict jobs.

We use RQ (Redis Queue) — minimal, zero broker beyond Redis itself,
plays nicely with FastAPI and Docker compose. Workers are launched as
their own container running `rq worker claim`.
"""

from __future__ import annotations

import os

from redis import Redis
from rq import Queue
from rq.job import Job

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "claim"

_redis = Redis.from_url(REDIS_URL)
_queue = Queue(QUEUE_NAME, connection=_redis)


def predict_task(text: str) -> dict:
    """Worker-side function — heavy import lives inside so the API process
    doesn't have to load PyTorch when QUEUE=1."""
    from app.predictor import Predictor

    # The Predictor is module-state on the worker side, loaded once per
    # worker process via a module-level cache.
    return _get_worker_predictor().predict(text).to_dict()


_worker_predictor = None


def _get_worker_predictor():
    global _worker_predictor
    if _worker_predictor is None:
        from app.predictor import Predictor

        _worker_predictor = Predictor()
    return _worker_predictor


def enqueue_predict(text: str) -> Job:
    return _queue.enqueue(predict_task, text, job_timeout=60, result_ttl=300)


def job_status(job_id: str) -> tuple[str, dict | None]:
    """Return (status_string, payload). status ∈ {queued, started, finished, failed, unknown}.

    Newer RQ versions return a JobStatus enum from get_status(); coerce
    to its lowercase string value so callers and templates can rely on
    plain strings."""
    try:
        job = Job.fetch(job_id, connection=_redis)
    except Exception:
        return "unknown", None
    raw = job.get_status(refresh=True)
    status = getattr(raw, "value", raw)  # enum -> "finished"; str passes through
    status = str(status).lower()
    payload = None
    if status == "finished":
        payload = job.result
    elif status == "failed":
        payload = {"error": str(job.exc_info or "job failed")}
    return status, payload
