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

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.predictor import Predictor

ROOT = Path(__file__).resolve().parent.parent
USE_QUEUE = os.environ.get("QUEUE", "0") == "1"
APP_URL = os.environ.get("APP_URL", "http://localhost:8000").rstrip("/")
API_URL = os.environ.get("API_URL", f"{APP_URL}/api").rstrip("/")

# Curl example for Swagger and the docs page — uses the configured API
# URL so when this is deployed to claims.jakehash.com the example
# automatically reflects that.
EXAMPLE_TEXT = "Inflation hit 9.1% in June 2022."
CURL_EXAMPLE = (
    f"curl -X POST {API_URL}/predict/sync "
    f"-H 'content-type: application/json' "
    f"-d '{{\"text\": \"{EXAMPLE_TEXT}\"}}'"
)

app = FastAPI(
    title="Claim Detection API",
    description=(
        "Sentence-level claim detection (fine-tuned Ettin-150m).\n\n"
        f"**Try it from your shell:**\n\n```\n{CURL_EXAMPLE}\n```\n\n"
        f"Browse the UI at {APP_URL}/."
    ),
    version="0.1.0",
    # The bare Swagger / ReDoc renderers stay reachable at /docs-raw and
    # /redoc-raw so they can be embedded in /docs and /redoc, which are
    # wrapped by our base template (preserving the nav and container width).
    docs_url="/docs-raw",
    redoc_url="/redoc-raw",
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
    text: str = Field(
        ...,
        min_length=1,
        description="Sentence to classify",
        examples=["Inflation hit 9.1% in June 2022."],
    )


class PredictResponse(BaseModel):
    is_claim: bool = Field(..., description="True if the text contains a check-worthy factual claim.")
    confidence: float = Field(..., description="Softmax probability of the predicted class. 0–1, higher is more confident.")
    label: str = Field(..., description="Either 'claim' or 'not_claim'.")


class EnqueueResponse(BaseModel):
    job_id: str = Field(..., description="Unique job id for this prediction.")
    stream_url: str = Field(..., description="Server-Sent-Events URL that streams `status` and final `result` events.")


# All public endpoints live under /api so the same domain can host the
# UI at / and the API at /api/*. The router is also mounted at the
# root for backward-compat; old paths return a 308 redirect to the
# /api equivalent (see `_install_legacy_redirects` below).
api = APIRouter(prefix="/api", tags=["claim-detection"])


@api.get(
    "/healthz",
    summary="Readiness probe",
    description="Returns `{status, queue}`. Use this for container healthchecks.",
)
def healthz() -> dict:
    return {"status": "ok", "queue": USE_QUEUE}


@api.post(
    "/predict/sync",
    response_model=PredictResponse,
    summary="Predict, blocking until result is ready (recommended)",
    description=(
        "Sends the sentence through the model and **waits for the result** before "
        "responding. Same payload shape whether the queue is enabled or not — when "
        "QUEUE=1 the request is internally enqueued and we poll until the worker "
        "finishes; when QUEUE=0 the predictor runs in-process. Bounded by "
        "`STREAM_TIMEOUT_SEC` (default 60s).\n\n"
        f"```\n{CURL_EXAMPLE}\n```"
    ),
    responses={
        200: {"description": "Prediction completed."},
        422: {"description": "Empty or malformed text."},
        503: {"description": "Model checkpoint not loaded."},
        504: {"description": "Worker took longer than the timeout."},
    },
)
async def predict_sync(req: PredictRequest):
    if not USE_QUEUE:
        # Sync mode: run predictor in-process. No queue involved.
        try:
            result = get_predictor().predict(req.text)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return PredictResponse(**result.to_dict())

    # Queued mode: enqueue + poll until done. Reuses the same status-poll
    # loop the SSE endpoint uses but returns JSON when finished.
    import asyncio as _asyncio

    from app.queue import enqueue_predict, job_status

    job = enqueue_predict(req.text)
    deadline = _asyncio.get_event_loop().time() + POLL_TIMEOUT_SEC
    while True:
        status, payload = job_status(job.id)
        if status == "finished":
            return PredictResponse(**payload)
        if status == "failed":
            raise HTTPException(
                status_code=500,
                detail=(payload or {}).get("error", "job failed"),
            )
        if _asyncio.get_event_loop().time() > deadline:
            raise HTTPException(status_code=504, detail="worker timeout")
        await _asyncio.sleep(POLL_INTERVAL_SEC)


@api.post(
    "/predict",
    response_model=None,
    summary="Predict, returning immediately (queued + SSE)",
    description=(
        "When `QUEUE=1`: enqueues the job and returns `{job_id, stream_url}` with HTTP 202. "
        "Open the `stream_url` to receive Server-Sent-Events as the worker progresses.\n\n"
        "When `QUEUE=0`: runs synchronously and returns the prediction immediately "
        "(same shape as `/predict/sync`).\n\n"
        "**For most callers, prefer `/predict/sync`** — it's a single round-trip and the "
        "queue is invisible to the client."
    ),
)
def predict(req: PredictRequest):
    if USE_QUEUE:
        from app.queue import enqueue_predict

        job = enqueue_predict(req.text)
        return JSONResponse(
            {"job_id": job.id, "stream_url": f"{API_URL}/predict/{job.id}/stream"},
            status_code=202,
        )

    try:
        result = get_predictor().predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**result.to_dict())


@api.get(
    "/predict/{job_id}/stream",
    summary="SSE stream for a queued job",
    description=(
        "Connect to receive `status` events (queued → started → finished/failed) and a "
        "final `result` event whose data is an HTML fragment ready to swap into the page. "
        "Closes automatically once the result event is sent."
    ),
)
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


POLL_INTERVAL_SEC = float(os.environ.get("STREAM_POLL_SEC", "0.05"))
POLL_TIMEOUT_SEC = float(os.environ.get("STREAM_TIMEOUT_SEC", "60"))


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


# Mount the API under /api/*. Also expose the same endpoints at the
# old paths via 308 redirects so existing clients (the local docker
# stack, the integration tests, scripts) keep working.
app.include_router(api)


def _install_legacy_redirects() -> None:
    """308-redirect /healthz, /predict, /predict/{id}/stream, /predict/sync
    to their /api/* equivalents. 308 preserves method + body."""
    legacy_paths = [
        ("/healthz", "/api/healthz", ["GET"]),
        ("/predict", "/api/predict", ["POST"]),
        ("/predict/sync", "/api/predict/sync", ["POST"]),
    ]
    for old, new, methods in legacy_paths:
        async def _redirect(request: Request, _new=new):
            target = _new + (("?" + request.url.query) if request.url.query else "")
            return RedirectResponse(url=target, status_code=308)
        app.add_api_route(old, _redirect, methods=methods, include_in_schema=False)

    # Job-id stream redirect needs the path param threaded through.
    async def _redirect_stream(job_id: str, request: Request):
        return RedirectResponse(url=f"/api/predict/{job_id}/stream", status_code=308)
    app.add_api_route(
        "/predict/{job_id}/stream", _redirect_stream, methods=["GET"], include_in_schema=False
    )


_install_legacy_redirects()


# ---------- HTMX UI ----------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "index.html",
        {"queue": USE_QUEUE, "app_url": APP_URL, "api_url": API_URL, "curl_example": CURL_EXAMPLE},
    )


@app.get("/docs", response_class=HTMLResponse)
def docs_page(request: Request) -> HTMLResponse:
    """Swagger UI wrapped in our base layout — keeps the nav and container
    width consistent across pages. The raw Swagger renderer stays
    available at /docs-raw for direct linking."""
    return templates.TemplateResponse(request, "api_docs.html", {})


@app.get("/api-docs", response_class=HTMLResponse)
def api_docs_redirect() -> RedirectResponse:
    """Friendly alias for /docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> RedirectResponse:
    """Browsers request /favicon.ico by convention even when there's a
    <link> tag. Redirect to the static path."""
    return RedirectResponse(url="/static/favicon.ico", status_code=301)


UI_MAX_BATCH = int(os.environ.get("UI_MAX_BATCH", "1000"))


def _split_lines(raw: str) -> list[str]:
    """Split textarea input on newlines, drop blank/whitespace-only lines.
    Used by /ui/predict to support pasting many sentences at once."""
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request, text: str = Form("")):
    """Returns one or more <tr> HTML fragments to be prepended to the
    results table. Supports multi-line paste: each non-blank line is
    enqueued as its own job. Capped at UI_MAX_BATCH (default 1000)
    lines per submit so a runaway paste can't DoS the worker.

    The placeholder-row OOB delete fragment is appended once at the
    end of the response — htmx removes the placeholder after the
    first batch lands; subsequent batches' OOB swaps no-op."""
    lines = _split_lines(text)

    if not lines:
        return templates.TemplateResponse(
            request, "_row_error.html",
            {"error": "Please enter a sentence."},
            status_code=400,
        )
    if len(lines) > UI_MAX_BATCH:
        return templates.TemplateResponse(
            request, "_row_error.html",
            {"error": f"Up to {UI_MAX_BATCH:,} sentences per submit, got {len(lines):,}."},
            status_code=400,
        )

    rows_html: list[str] = []

    if USE_QUEUE:
        from app.queue import enqueue_predict

        for line in lines:
            job = enqueue_predict(line)
            rows_html.append(
                templates.get_template("_row_streaming.html").render(
                    request=request, job_id=job.id, text=line
                )
            )
    else:
        try:
            predictor = get_predictor()
        except FileNotFoundError as e:
            return templates.TemplateResponse(
                request, "_row_error.html", {"error": str(e)}, status_code=503
            )
        for line in lines:
            result = predictor.predict(line)
            rows_html.append(
                templates.get_template("_row_sync.html").render(
                    request=request, text=line, result=result.to_dict()
                )
            )

    # OOB delete of the placeholder row — emit once at the end so it's
    # idempotent across single and batch submits. The streaming row
    # template also includes one, so this is belt-and-suspenders.
    rows_html.append('<tr id="placeholder-row" hx-swap-oob="delete"></tr>')

    return HTMLResponse("\n".join(rows_html))


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request) -> HTMLResponse:
    from app.results_view import build_view

    view = build_view()
    return templates.TemplateResponse(request, "results.html", view)
