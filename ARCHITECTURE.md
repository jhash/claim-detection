# Architecture

Two views of the system: the **runtime/inference stack** (what the
deployed Docker stack actually does on a request) and the **training
pipeline** (the offline path that produces the model checkpoints).

Both diagrams are [Mermaid](https://mermaid.js.org/) — render natively
on GitHub, in VS Code with the Markdown Preview, and in any modern
docs site. The same diagrams are also available as an Excalidraw file
([`architecture.excalidraw`](architecture.excalidraw)) if you want a
polished presentation-quality version that you can drag-and-drop into
<https://excalidraw.com> to edit.

---

## Runtime / inference stack

What happens when a user types a sentence at <https://localhost:8000/>
or `curl`s `POST /api/predict/sync`.

```mermaid
flowchart LR
    %% External entry points
    Browser([Browser<br/>HTMX 1.9])
    CLI([curl / CLI<br/>HTTP client])

    %% Reverse-proxy boundary (none for now, here for clarity)
    subgraph host["Host machine — Docker Compose / Swarm"]
        direction LR

        subgraph api_svc["api container<br/>(uvicorn + FastAPI)"]
            direction TB
            Routes["Routes<br/>GET /<br/>GET /results<br/>GET /docs<br/>POST /api/predict/sync<br/>POST /api/predict<br/>GET /api/predict/{id}/stream<br/>GET /api/healthz"]
            Templates["Jinja2 templates<br/>app/templates/*.html"]
            ResultsView["app/results_view.py<br/>reads results/*.json"]
        end

        subgraph worker_svc["worker container<br/>(rq SimpleWorker)"]
            direction TB
            Predictor["app/predictor.py<br/>Predictor singleton<br/>(load model once)"]
            HFModel{{"Fine-tuned Ettin-150m<br/>safetensors weights<br/>~574 MB resident"}}
        end

        subgraph redis_svc["redis container<br/>(redis:7-alpine)"]
            Queue[(RQ queue<br/>'claim')]
            Results[(Job results<br/>TTL 5 min)]
        end

        ModelVol[("Read-only volume<br/>./runs/ettin-150m-ft/final → /model")]
        ResultsVol[("Read-only volume<br/>./results → /app/results")]
    end

    Browser -- "POST /ui/predict<br/>(HTMX form)" --> Routes
    Browser -- "GET /<br/>GET /results<br/>GET /docs" --> Routes
    Browser -- "GET /api/predict/{id}/stream<br/>(SSE)" --> Routes
    CLI -- "POST /api/predict/sync<br/>(JSON)" --> Routes

    Routes -- "enqueue(text)" --> Queue
    Worker_Loop["Worker loop:<br/>1. BLPOP from queue<br/>2. predictor.predict(text)<br/>3. write result"] -.- worker_svc
    Queue -- "BLPOP" --> Worker_Loop
    Worker_Loop --> Predictor
    Predictor -- "loads from" --> ModelVol
    HFModel -.- Predictor
    Worker_Loop -- "store {is_claim, confidence}" --> Results

    Routes -- "poll job_status<br/>(50 ms async)" --> Results
    Routes -- "render HTML<br/>{is_claim, confidence}<br/>via SSE event" --> Browser
    Routes -- "JSON response" --> CLI

    ResultsView --> ResultsVol
    Routes -- "/results page" --> ResultsView

    classDef external fill:#fef3c7,stroke:#92400e,color:#92400e
    classDef container fill:#dbeafe,stroke:#1d4ed8,color:#1d4ed8
    classDef volume fill:#f3f4f6,stroke:#6b7280,color:#374151
    classDef redis fill:#fee2e2,stroke:#991b1b,color:#991b1b

    class Browser,CLI external
    class api_svc,worker_svc container
    class redis_svc redis
    class ModelVol,ResultsVol volume
```

### Request flow, plain English

1. User types a sentence in the browser → HTMX `POST /ui/predict`.
2. **api** container puts a job on the **redis** queue, returns an
   HTML row fragment that opens an SSE stream.
3. **worker** container's `BLPOP` returns the job. Predictor — already
   in-memory from container start — runs `tokenize → forward pass →
   softmax`. Result `{is_claim, confidence, label}` lands back in
   redis.
4. **api**'s SSE handler (50 ms async polling) sees the new result,
   emits an `event: result` HTML fragment.
5. HTMX swaps the result HTML into the page row in place. No JS
   parsing, no client-side rendering.

Round-trip wall clock under load: **~150 ms steady state** (90 ms
inference + 25 ms avg poll wait + 30 ms HTTP). Sync mode (`QUEUE=0`,
no redis): ~80 ms — predictor runs in-process in the api container.

For the `curl POST /api/predict/sync` path: same internal flow, but
the api endpoint blocks on the same status-poll loop and returns the
final JSON in one HTTP round-trip.

---

## Training & evaluation pipeline

The offline path that produces the checkpoints in `runs/`. Runs once
per model on the M4 Air; outputs feed the inference stack above.

```mermaid
flowchart TB
    subgraph datasets["datasets/  (vendored sources)"]
        Verita["verita-composite/ours/<br/>train.csv: 10,425<br/>test.csv: 2,607"]
        Combined["combined-v1/<br/>train.csv: 16,527<br/>test.csv: 4,132"]
        Claimify["claimify/<br/>6,490 sentences<br/>(LLM-generated, OOD)"]
        CT22["verita-composite/CheckThat/<br/>911 tweets<br/>(OOD)"]
    end

    subgraph registry["src/models.py — ModelSpec registry"]
        Specs["9 model specs:<br/>ettin-150m-ft<br/>ettin-150m-ft-combined<br/>ettin-150m-pretrained<br/>bert-base-{ft,pretrained}<br/>modernbert-base-{ft,pretrained}<br/>roberta-base-{ft,pretrained}"]
    end

    subgraph training["src/pipeline.py — fine-tune loop"]
        Tokenize["AutoTokenizer<br/>truncate 256 tokens"]
        Trainer["HF Trainer<br/>3 epochs, lr 5e-5<br/>MPS on M-series Mac"]
    end

    subgraph evaluation["src/evaluate.py — in-domain eval"]
        EvalLoop["argmax(logits)<br/>P/R/F1/Acc"]
    end

    subgraph ood_eval["src/evaluate_ood.py — OOD eval"]
        OODLoop["Same model<br/>vs Claimify + CT22"]
    end

    subgraph compare["src/compare.py + src/build_combined_split.py"]
        BuildSplit["build_combined_split.py<br/>concatenate + dedup + 80/20"]
        CompareScript["compare.py<br/>RESULTS.md generator"]
    end

    HFHub[("HuggingFace Hub<br/>jhu-clsp/ettin-encoder-150m<br/>google-bert/bert-base-uncased<br/>FacebookAI/roberta-base<br/>answerdotai/ModernBERT-base")]

    Runs[("runs/<slug>/final/<br/>safetensors checkpoints")]
    Results[("results/<slug>.json<br/>results/ood/<slug>.json")]
    ResultsMD[/"RESULTS.md<br/>(rendered comparison)"/]

    Verita --> BuildSplit
    Claimify --> BuildSplit
    BuildSplit --> Combined

    Specs --> Tokenize
    HFHub --> Tokenize
    Verita -- "default training data" --> Tokenize
    Combined -- "ettin-150m-ft-combined only" --> Tokenize
    Tokenize --> Trainer
    Trainer --> Runs

    Runs --> EvalLoop
    Verita -- "test split" --> EvalLoop
    Combined -- "test split" --> EvalLoop
    EvalLoop --> Results

    Runs --> OODLoop
    Claimify --> OODLoop
    CT22 --> OODLoop
    OODLoop --> Results

    Results --> CompareScript
    CompareScript --> ResultsMD

    Runs -. "served by" .-> InferenceStack[/"Inference stack<br/>(see diagram above)"/]

    classDef data fill:#dcfce7,stroke:#15803d,color:#15803d
    classDef pipeline fill:#dbeafe,stroke:#1d4ed8,color:#1d4ed8
    classDef artifact fill:#fef3c7,stroke:#92400e,color:#92400e
    classDef external fill:#f3f4f6,stroke:#6b7280,color:#374151

    class Verita,Combined,Claimify,CT22 data
    class Tokenize,Trainer,EvalLoop,OODLoop,BuildSplit,CompareScript pipeline
    class Runs,Results,ResultsMD artifact
    class HFHub,InferenceStack external
```

### Pipeline flow, plain English

1. **Vendor** training data into `datasets/` (one-time; everything
   normalized to `text,label,source` CSV).
2. **Pick** a model from the registry in `src/models.py`. Each spec
   says: HF id, fine-tune or frozen probe, optional override of the
   training corpus (`data_dir`).
3. **Train**: `python -m src.pipeline --model <slug>` pulls the
   pretrained weights from HF Hub, tokenizes the configured training
   set, runs HF `Trainer` for 3 epochs on MPS, saves the checkpoint
   under `runs/<slug>/final/`.
4. **Evaluate** in-domain: `python -m src.evaluate --model <slug>`
   loads the checkpoint, predicts on the matching test split, writes
   metrics to `results/<slug>.json`.
5. **Evaluate** OOD: `python -m src.evaluate_ood --model <slug>`
   predicts the same checkpoint against Claimify + CT22, writes to
   `results/ood/<slug>.json`.
6. **Compare**: `python -m src.compare` aggregates every JSON in
   `results/` into a markdown table with column-best bolding,
   in-domain table, OOD table, side-by-side vs the Bell paper.
7. **Serve**: the inference stack mounts `runs/ettin-150m-ft/final/`
   as `/model` and the predictor loads it once at worker boot.

The whole 8-model sweep (4 fine-tunes + 4 frozen probes + 1 combined-v1
re-run of Ettin) takes ~4 hours wall clock on the M4 Air, no cloud.

---

## Editing these diagrams

### Mermaid (this file)

Edit the `mermaid` code blocks above. To preview locally:

- **VS Code**: install the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)
  extension, then open this file with `Cmd+Shift+V`.
- **GitHub**: just commit and view on github.com — Mermaid is rendered
  natively in `.md` files.
- **CLI**: `npm i -g @mermaid-js/mermaid-cli && mmdc -i ARCHITECTURE.md -o arch.png`.

### Excalidraw

`architecture.excalidraw` is a JSON file. Three ways to edit:

- **In-browser**: drag the file into <https://excalidraw.com>. Runs
  fully client-side, nothing leaves your machine. Save back over the
  file when done.
- **VS Code**: install the
  [Excalidraw Editor](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor)
  extension and open `architecture.excalidraw` directly.
- **Self-host**: `docker run -p 80:80 excalidraw/excalidraw` — same
  app, served from your machine.
