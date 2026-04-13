# System Architecture

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DJANGO FRONTEND                              │
│         (Views / Templates / DRF / Django Channels)            │
│                     [Render — free tier]                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / REST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI RAG SERVICE                          │
│   /upload  /extract  /query  /query/stream  /health  /metrics  │
│                     [Render — free tier]                        │
└────┬──────────────┬──────────────┬──────────────┬──────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
  AWS S3        BYTEZ API      FAISS        MLFLOW SERVER
 (storage)    (inference +    (in-proc,    (tracking +
              evaluation)    index on S3)  S3 artifacts)
```

## Services

### 1. Django Frontend
- Serves the web UI (document upload, chat interface, session management)
- Calls the FastAPI RAG service via internal HTTP
- Handles streaming responses via Server-Sent Events (SSE) from FastAPI
- No ML logic — purely presentation and user session management

**Key decision:** Django does NOT run RAG logic. It is a thin client over FastAPI. This keeps concerns separated and the RAG service independently testable.

### 2. FastAPI RAG Service
The core backend. Handles:
- Receiving upload events from Django
- Triggering the extraction pipeline
- Serving `/query` and `/query/stream` endpoints
- Logging all queries and results to MLflow
- Running A/B prompt variant selection

### 3. AWS S3 (Single Source of Truth)
Everything persists to S3. No local disk dependencies.

```
s3://loandoc-bucket/
├── uploads/
│   └── {user_id}/{session_id}/original.pdf
├── extracted/
│   └── {session_id}/
│       ├── raw_text.txt
│       ├── cleaned_text.txt
│       └── validation_report.json
├── chunks/
│   └── {session_id}/
│       ├── chunks.json
│       └── chunk_embeddings.npy (FAISS index)
└── mlflow/
    └── artifacts/  (MLflow artifact store)
```

### 4. Bytez API
Used for two purposes:
- **Inference:** Send prompt → get LLM completion (Mistral, Llama3, or similar)
- **Evaluation:** Score answer quality (faithfulness, relevance, correctness) using an evaluator model

This means a single API account covers both inference and eval — no separate OpenAI/Groq key needed.

See `08_llm_bytez.md` for details.

### 5. FAISS
In-process vector index (not a separate service). Loaded into RAM when a query arrives. Serialized to S3 after indexing. Deserialized from S3 at query time.

No separate vector DB container. This keeps the deployment footprint minimal.

### 6. MLflow Server
Tracks all experiments: prompt variants, retrieval params, Bytez eval scores.

**Deployment config for Render:**
- Tracking backend: Render Postgres (free 1GB)
- Artifact store: S3 (`s3://loandoc-bucket/mlflow/artifacts/`)
- The MLflow server runs as a separate Render web service (always-on, free tier)

## Request Flows

### Upload Flow
```
1. User selects PDF in Django UI
2. Django calls POST /upload on FastAPI
3. FastAPI uploads PDF to S3: uploads/{user_id}/{session_id}/original.pdf
4. FastAPI triggers extraction pipeline (async background task)
5. Pipeline writes extracted/cleaned/chunked artifacts to S3
6. FastAPI returns session_id to Django
7. Django polls /status/{session_id} until pipeline complete
8. Django enables chat interface
```

### Query Flow
```
1. User submits question in Django chat UI
2. Django calls POST /query on FastAPI with {question, session_id}
3. FastAPI loads FAISS index from S3 for session_id
4. FastAPI embeds question (MiniLM, local)
5. FAISS returns top-k chunks
6. Prompt Router selects prompt variant (A/B logic)
7. FastAPI calls Bytez API with assembled prompt
8. FastAPI logs query + response + chunks to MLflow
9. FastAPI calls Bytez evaluator to score the response
10. MLflow logs eval scores
11. FastAPI returns answer to Django
12. Django renders answer in chat UI
```

## Technology Choices — Rationale

| Component | Choice | Why |
|---|---|---|
| LLM Inference | Bytez API | Pay-per-token, no idle GPU cost, open models |
| LLM Evaluation | Bytez API | Same account, LLM-as-judge pattern |
| Embeddings | MiniLM-L6-v2 (local) | Free, fast, 384-dim, no API cost |
| Vector Index | FAISS | In-process, no extra service, serializable |
| PDF Extraction | PyMuPDF | Free, local, handles standard PDFs well |
| Storage | AWS S3 | Single source of truth, cheap, durable |
| Experiment Tracking | MLflow | Industry standard, self-hostable |
| Backend | FastAPI | Async, OpenAPI docs, streaming support |
| Frontend | Django | Shows full-stack depth, DRF for API |
| Containers | Docker Compose | Reproducible local dev + Render deploy |
| Deployment | Render | Free tier, supports Docker, supports Postgres |

## What Is Deliberately Excluded

| Feature | Reason |
|---|---|
| Redis cache | Adds infra; simple dict cache is enough for portfolio scale |
| Kubernetes | Overkill; Render handles orchestration |
| Separate embedding API | MiniLM runs locally in milliseconds |
| Airflow | Replaced by FastAPI BackgroundTasks |
| Prometheus/Grafana | MLflow covers observability for this use case |
