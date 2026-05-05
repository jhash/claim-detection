"""FastAPI app exposing claim-detection inference.

Endpoints:
  GET  /              HTMX UI (form + SSE-streamed result)
  GET  /results       server-rendered comparison table
  GET  /healthz       readiness probe
  POST /predict       sync predict (when QUEUE=0) or enqueue (when QUEUE=1)
  GET  /predict/{job_id}/stream  SSE progress + final result
"""

from __future__ import annotations

import asyncio
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


@app.get("/predict/{job_id}/stream")
async def predict_stream(job_id: str, request: Request) -> StreamingResponse:
    if not USE_QUEUE:
        raise HTTPException(status_code=400, detail="queue is disabled (QUEUE=0)")
    from app.queue import job_status

    async def event_stream():
        last_status = None
        while True:
            if await request.is_disconnected():
                break
            status, payload = job_status(job_id)
            if status != last_status:
                yield _sse_event("status", {"status": status})
                last_status = status
            if status in ("finished", "failed"):
                yield _sse_event("result", payload)
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------- HTMX UI ----------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"queue": USE_QUEUE})


@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request, text: str = Form("")):
    if not text.strip():
        return templates.TemplateResponse(
            request, "_result.html",
            {"error": "Please enter a sentence."},
            status_code=400,
        )
    if USE_QUEUE:
        from app.queue import enqueue_predict

        job = enqueue_predict(text)
        return templates.TemplateResponse(
            request, "_streaming.html",
            {"job_id": job.id, "text": text},
        )
    try:
        result = get_predictor().predict(text)
    except FileNotFoundError as e:
        return templates.TemplateResponse(
            request, "_result.html", {"error": str(e)}, status_code=503
        )
    return templates.TemplateResponse(
        request, "_result.html",
        {"text": text, "result": result.to_dict()},
    )


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request) -> HTMLResponse:
    from app.results_view import build_view

    view = build_view()
    return templates.TemplateResponse(request, "results.html", view)
