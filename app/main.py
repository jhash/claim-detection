"""FastAPI app exposing claim-detection inference.

Endpoints:
  GET  /              HTMX UI (form + SSE-streamed result)
  GET  /results       server-rendered comparison table
  GET  /healthz       readiness probe
  POST /predict       sync predict (when QUEUE=0) or enqueue (when QUEUE=1)
  GET  /predict/{job_id}/stream  SSE progress + final result
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.predictor import Predictor

ROOT = Path(__file__).resolve().parent.parent
USE_QUEUE = os.environ.get("QUEUE", "0") == "1"

app = FastAPI(
    title="Claim Detection API",
    description="Sentence-level claim detection (fine-tuned Ettin-150m).",
    version="0.1.0",
)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Lazy-loaded so tests can mock the predictor without instantiating it at import.
_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def set_predictor(p: Optional[Predictor]) -> None:
    """Test hook — inject a mock or reset to None."""
    global _predictor
    _predictor = p


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Sentence to classify")


class PredictResponse(BaseModel):
    is_claim: bool
    confidence: float
    label: str


class EnqueueResponse(BaseModel):
    job_id: str
    stream_url: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "queue": USE_QUEUE}


@app.post("/predict", response_model=None)
def predict(req: PredictRequest):
    if USE_QUEUE:
        # Lazy import: only required when the queue is enabled, keeps the
        # FastAPI image working even when Redis isn't installed locally.
        from app.queue import enqueue_predict

        job = enqueue_predict(req.text)
        return JSONResponse(
            {"job_id": job.id, "stream_url": f"/predict/{job.id}/stream"},
            status_code=202,
        )

    try:
        result = get_predictor().predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**result.to_dict())


POLL_INTERVAL_SEC = float(os.environ.get("STREAM_POLL_SEC", "0.05"))
POLL_TIMEOUT_SEC = float(os.environ.get("STREAM_TIMEOUT_SEC", "60"))


@app.get("/predict/{job_id}/stream")
async def predict_stream(job_id: str, request: Request) -> StreamingResponse:
    if not USE_QUEUE:
        raise HTTPException(status_code=400, detail="queue is disabled (QUEUE=0)")
    import asyncio as _asyncio

    from app.queue import job_status

    async def event_stream():
        # Tight-poll Redis for status changes. We tried pub/sub but
        # redis-py doesn't guarantee the SUBSCRIBE ack lands before our
        # status check, so events fired between subscribe-call and
        # subscribe-confirm get silently lost. 50 ms polling gives ~25 ms
        # average latency — well below human perception — without that
        # class of race condition. ~20 lookups/sec on a single Redis
        # instance is negligible.
        last_status = None
        deadline = _asyncio.get_event_loop().time() + POLL_TIMEOUT_SEC
        while True:
            if await request.is_disconnected():
                return
            status, payload = job_status(job_id)
            if status != last_status:
                yield _sse_event("status", _render_status_html(status))
                last_status = status
            if status in ("finished", "failed"):
                yield _sse_event("result", _render_result_html(payload, status))
                return
            if _asyncio.get_event_loop().time() > deadline:
                yield _sse_event(
                    "result",
                    _render_result_html({"error": "timed out waiting for worker"}, "failed"),
                )
                return
            await _asyncio.sleep(POLL_INTERVAL_SEC)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event: str, data: str) -> str:
    # SSE data must be prefixed line-by-line; collapse any newlines so the
    # client sees one event no matter what.
    flat = data.replace("\n", " ").strip()
    return f"event: {event}\ndata: {flat}\n\n"


def _render_status_html(status: str) -> str:
    label = {"queued": "queued…", "started": "running…", "finished": "done", "failed": "error"}.get(status, status)
    return f'<span class="status status-{status}">{label}</span>'


def _render_result_html(payload, status: str) -> str:
    if status == "failed" or not isinstance(payload, dict) or "is_claim" not in payload:
        msg = (payload or {}).get("error", "job failed") if isinstance(payload, dict) else "job failed"
        return f'<span class="result error">error: {msg}</span>'
    is_claim = bool(payload["is_claim"])
    conf = float(payload["confidence"])
    cls = "claim" if is_claim else "not-claim"
    verdict = "CLAIM" if is_claim else "NOT A CLAIM"
    pct = f"{conf * 100:.1f}"
    return (
        f'<span class="result {cls}"><strong>{verdict}</strong> — {pct}%'
        f'<div class="conf-bar"><div style="width:{pct}%"></div></div></span>'
    )


# ---------- HTMX UI ----------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"queue": USE_QUEUE})


@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request, text: str = Form("")):
    """Returns a single <tr> HTML fragment to be appended to the results
    table on the index page. The fragment is self-contained — it includes
    its own SSE connection (queued mode) or its already-resolved result
    (sync mode)."""
    text = text.strip()
    if not text:
        return templates.TemplateResponse(
            request, "_row_error.html",
            {"error": "Please enter a sentence."},
            status_code=400,
        )
    if USE_QUEUE:
        from app.queue import enqueue_predict

        job = enqueue_predict(text)
        return templates.TemplateResponse(
            request, "_row_streaming.html",
            {"job_id": job.id, "text": text},
        )
    try:
        result = get_predictor().predict(text)
    except FileNotFoundError as e:
        return templates.TemplateResponse(
            request, "_row_error.html", {"error": str(e)}, status_code=503
        )
    return templates.TemplateResponse(
        request, "_row_sync.html",
        {"text": text, "result": result.to_dict()},
    )


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request) -> HTMLResponse:
    from app.results_view import build_view

    view = build_view()
    return templates.TemplateResponse(request, "results.html", view)
