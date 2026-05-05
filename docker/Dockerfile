# Multi-stage build: small final image, CPU-only torch.
# Same image runs as both the API and the RQ worker (different CMD).

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only torch first (much smaller than the default wheel).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu

COPY requirements.txt .
# Skip torch (already installed CPU-only above) and the heavy training-only
# extras the API doesn't need. Datasets/accelerate are kept off the runtime
# image — we don't train inside the container.
RUN grep -vE '^(torch|datasets|accelerate|pandas)' requirements.txt > runtime-requirements.txt \
    && pip install -r runtime-requirements.txt


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_PATH=/model

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY app/ ./app/
COPY src/ ./src/

# The model is mounted as a volume in compose; copy the directory marker so
# import paths work even if no volume is mounted (image is then non-functional
# for inference but still parses).
RUN mkdir -p /model && chown -R app:app /model /app

USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
