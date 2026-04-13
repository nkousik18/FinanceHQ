# Build Roadmap

## Guiding Principle

Build in vertical slices — each phase produces something runnable end-to-end. No phase leaves you with half-built components that can't be tested.

---

## Phase 1 — Core Pipeline (No UI, No LLM)

**Goal:** PDF in → FAISS index out. Verify the data pipeline works before touching anything else.

**Deliverables:**
- [ ] New repo structure (clean — no old GCP/vLLM code)
- [ ] `fastapi_service/` skeleton with `/health` endpoint
- [ ] S3 client wrapper (`app/storage/s3_client.py`)
- [ ] Extraction: `app/pipeline/extractor.py` (PyMuPDF)
- [ ] Validation: `app/pipeline/validator.py`
- [ ] Cleaning: `app/pipeline/cleaner.py`
- [ ] Chunking: `app/pipeline/chunker.py`
- [ ] Embedding: `app/pipeline/embedder.py` (MiniLM singleton)
- [ ] FAISS index builder: `app/pipeline/indexer.py`
- [ ] Pipeline orchestrator: `app/pipeline/pipeline.py` (calls stages in order)
- [ ] FastAPI endpoint: `POST /upload` (triggers pipeline)
- [ ] FastAPI endpoint: `GET /sessions/{session_id}/status`
- [ ] Docker image for FastAPI builds and runs locally

**Test:** Upload a PDF via `curl`, check S3 for `chunks.json` and `faiss.index`.

---

## Phase 2 — Retrieval + Bytez Inference (No Django, No MLflow)

**Goal:** Question in → answer out. Core RAG loop working.

**Deliverables:**
- [ ] FAISS loader with in-memory cache (`app/retrieval/retriever.py`)
- [ ] Prompt router (`app/prompts/router.py`) — all 5 intents
- [ ] All prompt templates (`app/prompts/templates/`)
- [ ] Bytez client (`app/llm/bytez_client.py`) — non-streaming
- [ ] Bytez streaming client — streaming via async generator
- [ ] Token budget manager (`app/prompts/token_budget.py`)
- [ ] FastAPI endpoint: `POST /query` (non-streaming)
- [ ] FastAPI endpoint: `POST /query/stream` (streaming)
- [ ] Basic JWT auth (`app/auth/`) — register, login, token

**Test:** `curl -X POST /query` with a real question about an uploaded document. Get a real answer.

---

## Phase 3 — MLflow + Evaluation + A/B Testing

**Goal:** Every query tracked. A/B test infrastructure running.

**Deliverables:**
- [ ] MLflow server Docker image
- [ ] MLflow server running in Docker Compose
- [ ] Bytez evaluator client (`app/llm/evaluator.py`)
- [ ] Query tracker (`app/tracking/tracker.py`) — background task
- [ ] A/B variant selector (`app/tracking/ab_selector.py`)
- [ ] Experiment config file (`experiment_config.json`)
- [ ] All MLflow params/metrics/artifacts logged per query (per spec in `10_mlflow_tracking.md`)
- [ ] Pipeline runs logged to `loandoc_pipeline_runs` experiment

**Test:** Make 10 queries. Open MLflow UI at `localhost:5000`. Verify all runs appear with correct params and metrics. Verify A and B variants are both assigned.

---

## Phase 4 — Django Frontend

**Goal:** Working web UI end-to-end.

**Deliverables:**
- [ ] Django project scaffold (`django_frontend/`)
- [ ] Django Postgres models and migrations (`DocumentSession`)
- [ ] Accounts app: signup, login, logout
- [ ] Documents app: upload view, status polling view, sessions list
- [ ] Chat app: chat view, SSE streaming view
- [ ] HTML templates (clean, functional — no need for a design system yet)
- [ ] `chat.js` — EventSource handler for streaming
- [ ] Django added to Docker Compose
- [ ] Django calling FastAPI internally (internal Docker Compose URL)

**Test:** Full flow in browser: sign up → upload PDF → wait for READY → ask a question → see streaming response.

---

## Phase 5 — Deployment to Render

**Goal:** Live on the internet. Shareable link.

**Deliverables:**
- [ ] `render.yaml` configured for all 3 services + DB
- [ ] All `.env*` files in `.gitignore`
- [ ] All secrets set in Render Dashboard
- [ ] Django migrations run on Render
- [ ] MLflow server alive and accessible
- [ ] UptimeRobot monitoring MLflow
- [ ] GitHub Actions CI/CD workflow (test → deploy)
- [ ] README with live URL, architecture diagram, demo instructions

**Test:** Share the URL with someone not on your machine. They can sign up, upload a PDF, and get answers.

---

## Phase 6 — Polish and Portfolio Presentation

**Goal:** The project looks as impressive as it actually is.

**Deliverables:**
- [ ] Architecture diagram (draw.io or Excalidraw — add to `docs/` and README)
- [ ] README with: what it is, architecture, live demo link, how to run locally, key design decisions
- [ ] MLflow screenshot in README showing A/B test results with real data
- [ ] Cost breakdown section in README (actual S3 + Bytez costs)
- [ ] At least one A/B experiment concluded with a winner variant documented
- [ ] Cleaned repo: remove `home/` directory, remove dead GCP/vLLM code, clean up `.gitignore`

---

## What to Build First When Starting from Scratch

### New Directory Structure

```
loan-doc-ai-code/          (existing repo)
├── docs/                  (this folder — already done)
├── fastapi_service/       (RAG backend — build in Phase 1)
│   ├── app/
│   │   ├── pipeline/
│   │   ├── retrieval/
│   │   ├── prompts/
│   │   ├── llm/
│   │   ├── auth/
│   │   ├── tracking/
│   │   ├── storage/
│   │   └── server.py
│   └── tests/
├── django_frontend/       (web UI — build in Phase 4)
│   ├── config/
│   ├── accounts/
│   ├── documents/
│   ├── chat/
│   └── static/
├── docker/
│   ├── fastapi/Dockerfile
│   ├── django/Dockerfile
│   └── mlflow/Dockerfile
├── docker-compose.yml
├── render.yaml
├── .github/workflows/deploy.yml
├── requirements.fastapi.txt
├── requirements.django.txt
├── .gitignore
└── README.md
```

The old folders (`llm-microservice/`, `scripts/`, `airflow/`, `evaluation/`, `home/`) can remain for now as reference and are deleted in Phase 6 cleanup.

---

## Decision Log

Decisions made in the architecture phase, not to be revisited during build:

| Decision | Rationale |
|---|---|
| PyMuPDF over Textract | Free, no cross-cloud complexity, sufficient for digital PDFs |
| FAISS over ChromaDB | No Docker bloat, in-process, serializable to S3 |
| Bytez for both inference and eval | Single API account, no separate OpenAI/Groq key |
| Django over Streamlit | Shows full-stack depth for job search |
| SSE over WebSockets | Simpler than Django Channels, sufficient for one-way streaming |
| MLflow over custom logging | Industry standard, self-hostable, built-in UI |
| Render over Railway/Fly.io | Docker-native free tier, auto-deploy from GitHub |
| Sliding window chunking over semantic | Deterministic, fast, sufficient for structured loan docs |
| MiniLM over API embeddings | $0 cost, 384-dim is enough for this domain |
