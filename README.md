# FinanceHQ — Loan Document RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for loan documents. Upload a PDF loan application, ask natural-language questions, get accurate document-grounded answers.

Built as a portfolio project demonstrating the full MLOps lifecycle: document ingestion, vector retrieval, LLM inference, A/B prompt testing, experiment tracking, and cloud deployment.

---

## What It Does

- Accepts PDF loan documents (applications, agreements, mortgage forms)
- Extracts structured data using AWS Textract — handles multi-column layouts, handwritten fields, tables, and form key-value pairs
- Cleans, validates, and converts extracted content to a structured Markdown document
- Chunks and embeds the document using MiniLM (local, free)
- Stores a FAISS vector index in S3
- Answers natural-language questions using RAG — retrieves relevant chunks, assembles a prompt, calls an LLM via Bytez API
- Tracks every query, prompt variant, and evaluation score in MLflow
- Runs A/B tests on prompt templates to find what works best

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                            │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DJANGO FRONTEND                               │
│          Upload UI · Chat UI · Session management               │
│                    [Render — free tier]                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │ REST
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FASTAPI RAG SERVICE                           │
│   /upload · /query · /query/stream · /health · /metrics         │
│                    [Render — free tier]                          │
└────┬──────────────┬──────────────┬──────────────┬───────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
  AWS S3        BYTEZ API      FAISS        MLFLOW SERVER
 (storage)   (LLM inference  (in-process,  (experiment
              + evaluation)  index on S3)   tracking)
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| AWS Textract over PyMuPDF | Handles handwriting, multi-column, tables — critical for real loan forms |
| TABLES + FORMS feature types | 5.5/10 → 8.5/10 extraction quality; structured key-value pairs for form fields |
| DSPy as LLM wrapper | Clean, modular prompt signatures; easy to swap models |
| FAISS in-process | No extra service; serialized to S3; zero idle cost |
| MiniLM embeddings (local) | Free, fast, 384-dim — sufficient for document retrieval |
| Bytez API for inference + eval | Single API account covers both; pay-per-token, no idle GPU |
| MLflow for tracking | Industry standard; self-hostable; built-in A/B experiment UI |
| SSE over WebSockets | Simpler than Django Channels; sufficient for one-way streaming |
| Render over AWS ECS | Docker-native free tier; auto-deploy from GitHub |

---

## Project Structure

```
FinanceHQ/
├── fastapi_service/               # RAG backend
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic settings from .env
│   │   │   └── logging.py         # Structlog setup (pretty dev / JSON prod)
│   │   ├── storage/
│   │   │   ├── s3_client.py       # S3 wrapper — upload/download/exists/list
│   │   │   └── keys.py            # All S3 key paths in one place
│   │   ├── pipeline/
│   │   │   ├── extractor.py       # AWS Textract (sync + async, TABLES + FORMS)
│   │   │   ├── cleaner.py         # Number normalisation, header removal, dedup
│   │   │   ├── validator.py       # Confidence checks, required fields, handwriting flags
│   │   │   └── markdown_writer.py # Clean .md output from extraction results
│   │   ├── retrieval/             # FAISS loader + query (Phase 2)
│   │   ├── prompts/               # DSPy signatures + prompt templates (Phase 2)
│   │   ├── llm/                   # Bytez client via DSPy (Phase 2)
│   │   ├── tracking/              # MLflow tracker + A/B selector (Phase 3)
│   │   └── auth/                  # JWT auth (Phase 2)
│   └── tests/
│       ├── pipeline/
│       │   ├── test_extractor.py
│       │   ├── test_cleaner.py
│       │   └── test_validator.py
│       └── test_s3_client.py
├── django_frontend/               # Web UI (Phase 4)
├── docker/                        # Dockerfiles (Phase 4)
├── scripts/
│   ├── setup_s3_bucket.py         # One-time S3 bucket creation
│   ├── test_textract.py           # Explore Textract output on a real PDF
│   └── test_clean_validate.py     # Run cleaner + validator locally (no AWS)
├── docs/                          # Architecture and design docs
├── requirements.fastapi.txt
├── .env.example
└── .gitignore
```

---

## S3 Layout

```
s3://{bucket}/
├── uploads/{session_id}/original.pdf
├── extracted/{session_id}/
│   ├── raw_text.txt
│   ├── cleaned_text.txt
│   ├── validation_report.json
│   └── textract_response.json     ← raw Textract output (audit trail)
├── chunks/{session_id}/
│   ├── chunks.json
│   ├── embeddings.npy
│   └── faiss.index
└── mlflow/artifacts/              ← managed by MLflow
```

---

## Extraction Pipeline (Phase 1 — Complete)

### What Textract extracts

- **Text blocks** — all LINE and WORD blocks with confidence scores
- **Form fields** — KEY_VALUE_SET pairs: `Loan Amount → 7,50,000`, `Tenure → 36 MONTHS`
- **Tables** — full row/column structure: income details, liability schedules, document checklists
- **Handwriting** — detected and flagged per page (TextType = HANDWRITING)

### Cleaning steps

| Issue | Example | Fix |
|---|---|---|
| Spaces in numbers | `110 000` | → `110000` |
| Indian lakh formatting | `55 00 000` | → `5500000` |
| Digit-letter concat | `36MONTHS` | → `36 MONTHS` |
| Repeated headers | `Shinhan Bank` on every page | Stripped |
| Empty checkbox fields | `Male →` (empty) | Removed |
| Duplicate keys | Same field extracted twice | Deduplicated, keep highest-confidence non-empty |
| Truncated values | `15/02/200` | Flagged in validation report |

### Validation checks

| Check | Threshold | Action |
|---|---|---|
| Overall confidence | < 70% = error, < 85% = warning | Block or flag |
| Per-page confidence | Same thresholds | Per-page flag |
| Required fields present | 10 loan-specific fields | Warning if missing |
| Handwriting detected | Any page | Info flag |
| Truncated values | Pattern matched | Warning per field |
| No structured data | Zero fields + zero tables | Error, block |

### Output

A structured Markdown file saved to S3:
- Extraction summary (confidence, pages, handwriting)
- Validation report (errors, warnings)
- Key loan fields table
- All form fields table
- All tables rendered as Markdown
- Full page-by-page text

---

## Local Setup

### Prerequisites

- Python 3.11+
- AWS account with Textract + S3 access
- `financehq-dev` IAM user with `FinanceHQPolicy` attached

### Install

```bash
git clone <repo>
cd FinanceHQ
python -m venv venv
source venv/bin/activate
pip install -r requirements.fastapi.txt
```

### Configure

```bash
cp .env.example .env
# Fill in: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
```

### Create S3 bucket

```bash
python scripts/setup_s3_bucket.py
```

### Test extraction on a PDF

```bash
# Upload + extract via Textract
python scripts/test_textract.py path/to/loan.pdf

# Clean + validate locally (no AWS call — reads saved JSON)
python scripts/test_clean_validate.py scripts/output/loan1_raw.json
```

Output markdown saved to `scripts/output/loan1_cleaned.md`.

### Run tests

```bash
cd fastapi_service
pytest tests/ -v
```

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Extraction Pipeline | ✅ Complete | PDF → Textract → clean → validate → Markdown |
| 2 — RAG Query | 🔲 Next | FAISS retrieval + DSPy + Bytez LLM inference |
| 3 — MLflow + A/B | 🔲 Planned | Query tracking, prompt variants, eval scores |
| 4 — Django Frontend | 🔲 Planned | Upload UI, chat interface, session management |
| 5 — Render Deployment | 🔲 Planned | Live URL, CI/CD, monitoring |
| 6 — Polish | 🔲 Planned | Architecture diagram, cost breakdown, demo |

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret | Yes |
| `AWS_REGION` | AWS region (default: us-east-1) | No |
| `S3_BUCKET` | S3 bucket name | Yes |
| `BYTEZ_API_KEY` | Bytez API key for LLM inference | Phase 2 |
| `MLFLOW_TRACKING_URI` | MLflow server URL | Phase 3 |
| `LOG_LEVEL` | INFO / DEBUG / WARNING | No |
| `ENVIRONMENT` | development (pretty logs) / production (JSON) | No |

---

## Tech Stack

| Component | Technology |
|---|---|
| PDF Extraction | AWS Textract (TABLES + FORMS) |
| LLM Inference | Bytez API via DSPy |
| LLM Evaluation | Bytez API (same account) |
| Embeddings | MiniLM-L6-v2 (local) |
| Vector Index | FAISS (in-process) |
| Storage | AWS S3 |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Frontend | Django |
| Logging | structlog |
| Deployment | Render (free tier) |
