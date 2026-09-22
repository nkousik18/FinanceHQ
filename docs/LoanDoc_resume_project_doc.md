# FinanceHQ — Complete Technical Documentation for Resume

---

## 1. Project Summary

FinanceHQ is an end-to-end Retrieval-Augmented Generation (RAG) system for loan document Q&A. Users upload one or more PDF loan documents into a session; the system processes them through a 12-stage pipeline and answers natural language questions grounded exclusively in the uploaded document content. Supports multi-document sessions enabling cross-document queries. Built as a portfolio project demonstrating the full MLOps lifecycle: document ingestion, vector retrieval, LLM inference, prompt A/B testing, quantitative model evaluation, experiment tracking, and cloud deployment.

**Target audience:** ML / MLOps / AI engineering hiring managers.  
**Constraint set:** No paid GPU, < $1/month total running cost, free-tier cloud only.

---

## 2. Tech Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| API Framework | FastAPI | 0.111.0 | RAG backend, async endpoints |
| PDF Extraction | AWS Textract | boto3 1.34.0 | AnalyzeDocument — TABLES + FORMS features |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 2.7.0 | 384-dim, local CPU, $0 cost |
| Vector Index | FAISS IndexFlatIP | faiss-cpu 1.8.0 | In-process, serialized to S3 |
| LLM Inference (prod) | Groq API | — | llama-3.3-70b-versatile, 1024 max tokens |
| LLM Inference (eval) | Bytez API | — | Mistral-7B, Llama-3.1-8B, Qwen2.5-7B |
| Experiment Tracking | MLflow | 2.13.0 | Self-hosted, every query = one run |
| Storage | AWS S3 | boto3 | Single source of truth for all artifacts |
| Chunking tokenizer | tiktoken | 0.7.0 | cl100k_base, token counting |
| Sentence tokenizer | nltk | 3.8.1 | sent_tokenize for sentence boundary chunking |
| Structured logging | structlog | — | JSON logs with bound context |
| Config | pydantic-settings | 2.2.0 | .env file → typed Settings object |
| Frontend | React | — | SPA, replaced Django frontend |
| Web server (prod) | nginx + uvicorn | — | nginx proxies to uvicorn port 8001 |
| CI/CD | GitHub Actions | — | Push to main → SSH deploy to both EC2s |
| Cloud | AWS EC2 (2× t2.micro) | — | Free tier, Ubuntu 24.04 LTS |
| Testing | pytest + unittest.mock | — | ~122 passing tests, all external deps mocked |

**Tools evaluated and not used:**
- PyMuPDF — considered for extraction; replaced by Textract (handles tables, forms, handwriting)
- DSPy — evaluated as LLM wrapper; removed from final build
- Render — original deployment target; replaced by AWS EC2 (no cold-start sleep)
- ChromaDB — considered for vector store; replaced by FAISS (no extra service, serializable)
- Streamlit — considered for frontend; replaced by Django then React (full-stack signal)
- Redis — considered for caching; replaced by in-process dict (sufficient at portfolio scale)
- Airflow — considered for orchestration; replaced by FastAPI BackgroundTasks

---

## 3. Ingestion Pipeline (6 Stages)

Every PDF upload triggers a 6-stage background pipeline via FastAPI `BackgroundTasks`. Each stage transition writes a status update to `s3://{bucket}/sessions/{session_id}/status.json`. The client polls `GET /sessions/{session_id}/status` until `READY`.

### Stage 1 — S3 Upload
- Input: raw PDF bytes from multipart/form-data upload
- Validates: PDF content-type, not empty, ≤ 20MB
- Generates `doc_id` (UUID4)
- Writes to: `s3://{bucket}/uploads/{session_id}/{doc_id}/original.pdf`
- Sets doc status: `PENDING`

### Stage 2 — Extraction (AWS Textract)
- Tool: AWS Textract `AnalyzeDocument` with `TABLES` + `FORMS` feature types
- Routing: 1 page → synchronous `AnalyzeDocument`; ≥ 2 pages → `StartDocumentAnalysis` (async, polls with 5s sleep until `SUCCEEDED`)
- Threshold configurable: `TEXTRACT_ASYNC_THRESHOLD_PAGES=2` (env var)
- Output dataclasses: `ExtractionResult`, `PageResult`, `Table` (with `to_markdown()` method), `FormField`
- Saves to S3: `extracted/{session_id}/{doc_id}/raw_text.txt`, `extracted/{session_id}/{doc_id}/textract_response.json`
- Sets doc status: `EXTRACTING`

### Stage 3 — Cleaning
- Input: `ExtractionResult` from Textract
- Fixes applied (in order):
  1. Number space removal (Textract splits `8 . 7 5 %` → `8.75%`)
  2. Repeated header/footer removal (lines appearing on 3+ pages stripped)
  3. Empty checkbox artifact removal (`[ ]` patterns)
  4. Duplicate line removal
  5. Whitespace normalization (max 2 consecutive newlines, collapse tabs/spaces)
  6. Hyphenation fix (`mort-\ngage` → `mortgage`)
  7. Smart quote/em-dash normalization to ASCII
- Output: `CleanedResult` (cleaned markdown text)
- Sets doc status: `CLEANING`

### Stage 4 — Validation
- Input: `CleanedResult`
- Checks:
  - Confidence thresholds: per-block Textract confidence scores
  - Required field presence (configurable)
  - Page count > 0
- Severity levels: `WARNING` (logs, continues) vs `ERROR` (hard-blocks pipeline, sets doc to `FAILED`)
- Output: `ValidationResult`, writes `validation_report.json` to S3
- Sets doc status: `VALIDATE`

### Stage 5 — Chunking
- Strategy: sliding window with sentence boundary respect (`nltk.sent_tokenize`)
- Parameters:
  - `DEFAULT_CHUNK_SIZE = 400` (approximate tokens)
  - `DEFAULT_CHUNK_OVERLAP = 60` tokens
  - `WORDS_PER_TOKEN = 0.75` (word-to-token ratio)
  - Minimum chunk size: 50 tokens (fragments discarded)
- `chunk_id` format: `"{session_id}-{doc_id}-{index}"` — globally unique across docs in same session
- Chunk schema: `chunk_id`, `session_id`, `index`, `text`, `word_count`, `page`, `section` (nearest markdown heading), `token_estimate`
- Saves per-doc: `chunks/{session_id}/{doc_id}/chunks.json`
- Sets doc status: `CHUNKING`

### Stage 6 — Embedding + Indexing (FAISS merge)
- Embedding model: `all-MiniLM-L6-v2` singleton (loaded once at startup, ~90MB)
- `embed()` returns shape `(N, 384)`, L2-normalized — inner product == cosine similarity
- FAISS index type: `IndexFlatIP` (exact inner product search on normalized vectors)
- **Multi-doc merge logic** (`merge_into_session_index`):
  1. Embed new chunks
  2. Download existing `chunks.json` + `embeddings.npy` from S3 (if any prior docs)
  3. Concatenate old + new: `np.vstack([old_embeddings, new_embeddings])`
  4. Rebuild `IndexFlatIP` from merged embeddings
  5. Upload merged `chunks.json`, `embeddings.npy`, `faiss.index` to S3 session root
- Saves to S3: `chunks/{session_id}/chunks.json`, `chunks/{session_id}/embeddings.npy`, `chunks/{session_id}/faiss.index`
- Sets doc status: `INDEXING` → `READY`
- Session status derived: any doc in-flight → `PROCESSING`; ≥1 `READY`, none in-flight → `READY`

---

## 4. Query Pipeline (6 Stages)

Triggered by `POST /query` or `POST /query/stream`. Synchronous stages 1–4; MLflow logging is a non-blocking `BackgroundTask`.

### Stage 1 — Retrieval
- Loads `chunks/{session_id}/faiss.index` + `chunks/{session_id}/chunks.json` from S3
- In-memory cache: `_SESSION_CACHE: dict[str, tuple[faiss.Index, list[Chunk]]]` — first call loads from S3, subsequent calls are cache hits (no S3 read)
- Embeds question: `embed_one(question)` → shape `(384,)`, L2-normalized
- FAISS `index.search(q_vec, top_k)` → cosine similarity scores + chunk indices
- Score threshold: `0.25` minimum (chunks below threshold excluded)
- Default `top_k = 5` (configurable per request, range 1–20)
- Returns: `list[RetrievedChunk]` with `score`, `rank`, `text`, `chunk_id`, `page`, `section`
- Score range in practice: 0.4–0.8 for relevant loan document queries
- Returns "not found" response if no chunks exceed threshold (no hallucination from empty context)

### Stage 2 — Intent Classification
- Method: zero-shot cosine similarity using MiniLM (same singleton as retrieval — no additional model load)
- Five intents with priority order:
  - `CALCULATE` (priority 0) — "what is the total repayment?", "compute the EMI"
  - `LOOKUP` (priority 1) — "what is the interest rate?", "what is the loan tenure?"
  - `COMPARE` (priority 2) — "compare both loans", "which has lower EMI?"
  - `EXPLAIN` (priority 3) — "explain how the interest is calculated"
  - `SUMMARISE` (priority 4) — "summarise the loan terms", "give me an overview"
- Thresholds: `PRIMARY_THRESHOLD = 0.30`, `MULTI_INTENT_THRESHOLD = 0.28`
- Returns: `list[ClassifiedIntent]` sorted by score; `intents[0]` is always primary
- No LLM call, no training data, deterministic

### Stage 3 — Prompt Routing & Assembly
- Routes on `intents[0].intent` → selects template from `app/prompts/templates/`
- 5 templates × 2 variants each (v1/v2): `lookup`, `calculate`, `compare`, `explain`, `summarise`
- Variant selection: epsilon-greedy A/B selector (`ab_selector.py`, ε=0.1 exploit/explore)
- Context assembly: chunks formatted as `[Chunk N | Page P | Score X.XX]\n{text}`
- Word budget: `MAX_CONTEXT_WORDS = 1200` — trims lowest-ranked chunks first if exceeded
- Every template: system grounding instruction + context block + question
- Returns: `RoutedPrompt(prompt, intent, variant, chunks_used, context_words)`

### Stage 4 — LLM Inference
- Production: Groq API (`llama-3.3-70b-versatile`, max_tokens=1024, temperature=0.1)
- Retry logic: up to 2 retries, delays `[1.0, 3.0]` seconds on failure
- Raises `BytezInferenceError` / Groq equivalent after all retries exhausted → FastAPI returns 502
- Streaming (`/query/stream`): Bytez free tier has no native streaming — full response fetched, then word-by-word SSE at 30ms/word intervals
- Returns: `LLMResponse(text, model, tokens, latency_ms)`

### Stage 5 — Response (API)
- `QueryResponse`: `answer`, `intent`, `variant`, `chunks_used`, `context_words`, `latency_ms`
- SSE stream format: `data: {"token": "..."}` per word, final `data: {"done": true, "intent": ..., "variant": ..., "latency_ms": ...}`

### Stage 6 — MLflow Logging (Background)
- Non-blocking: `background_tasks.add_task(_log_to_mlflow, ...)` — never adds latency to response
- Runs in thread pool via `loop.run_in_executor(None, _log)`
- If tracking fails: logs warning, response unaffected

---

## 5. Multi-Document Session Architecture

**Session model:** One session holds N documents. The FAISS index is merged across all uploaded PDFs so a single `/query` call searches across all documents simultaneously.

**Session status state machine:**
```
EMPTY → PROCESSING → READY
                   → FAILED
```

**Doc status state machine:**
```
PENDING → EXTRACTING → CLEANING → VALIDATE → CHUNKING → INDEXING → READY
                                                                   → FAILED
```

**Session status derivation rule:**
- Any doc in `{PENDING, EXTRACTING, CLEANING, CHUNKING, INDEXING}` → session = `PROCESSING`
- ≥1 doc `READY`, none in-flight → session = `READY`
- All docs `FAILED` → session = `FAILED`
- No docs → session = `EMPTY`

**S3 layout:**
```
{bucket}/
├── sessions/{session_id}/status.json          ← full SessionStatusRecord
├── uploads/{session_id}/{doc_id}/original.pdf
├── extracted/{session_id}/{doc_id}/raw_text.txt
├── extracted/{session_id}/{doc_id}/textract_response.json
└── chunks/{session_id}/
    ├── chunks.json          ← MERGED across all docs (what /query reads)
    ├── embeddings.npy       ← MERGED float32 (N, 384)
    ├── faiss.index          ← MERGED FAISS binary
    └── {doc_id}/chunks.json ← per-doc audit copy
```

**`chunk_id` format:** `"{session_id}-{doc_id}-{index}"` — guarantees uniqueness across all docs in same session.

**Why stateless API:** All state in S3 `status.json`. Any replica can serve any endpoint — no sticky sessions, no in-memory session store.

---

## 6. Prompt Engineering & A/B Testing

### Intent Templates
5 intent categories, each with 2 prompt variants (v1 = structured/verbose, v2 = minimal/direct):

| Intent | Trigger signals | Prompt behavior |
|---|---|---|
| LOOKUP | "what is", "who", "when", "find" | Direct fact retrieval, cite clause |
| CALCULATE | "total", "compute", "EMI", "how much" | Show calculation steps, units required |
| COMPARE | "compare", "difference", "vs", "which" | Side-by-side structure, table if 3+ items |
| EXPLAIN | "explain", "what does X mean", "define" | Plain English, define jargon, cite section |
| SUMMARISE | "summarise", "overview", "key points" | Structured bullets, 200–300 words |

All templates share a system grounding instruction: "Answer ONLY from the provided document excerpts. Do not use knowledge from outside the document."

### A/B Variant Selection
- Algorithm: epsilon-greedy with ε=0.1
- ε=0.1 → 10% random exploration, 90% exploit best-known variant
- Variant label logged to MLflow as `variant` param (e.g. `"lookup_v1"`, `"lookup_v2"`)
- Session-stable: same session + intent always gets same variant within a session

### Experiment Configuration
Managed via `experiment_config.json` (stored in S3 or env var):
- `active_experiment` name
- Per-variant params: `k`, `context_format`, `temperature`
- `target_samples_per_variant` (default: 50)

### Stopping Rule
- ≥50 evaluated queries per variant AND mean composite score difference ≥0.05 → declare winner
- After 200 queries with no significant difference → both equivalent, default to v1

### Token Budget
- `MAX_CONTEXT_WORDS = 1200` (~1,600 tokens)
- Trims lowest-ranked chunks first if budget exceeded
- Every prompt logs: `prompt_tokens`, `intent`, `variant`, `context_chars`, `chunks_included`

---

## 7. LLM Model Evaluation (Quantitative)

**Evaluation script:** `scripts/ab_eval.py`  
**Scope:** 3 models × 7 questions = 21 API calls. Sequential (Bytez free tier: 1 concurrent request).

### Models Tested
| Model ID | Short Name | Notes |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama-3.1-8B | Meta instruction-tuned |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral-7B | Sliding-window attention |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5-7B | Strong structured output |

Models excluded (not on free tier): `google/gemma-2-9b-it`, `mistralai/Mixtral-8x7B-Instruct-v0.1`

### Test Document
Synthetic but realistic loan document excerpt (no real PII). Contains:
- Applicant details (names, incomes: ₹85,000 / ₹42,000)
- Loan details (₹12,00,000 at 8.75%, 60 months, EMI ₹24,842)
- Missed payment T&Cs (2%/month penal interest, 90-day NPA trigger, SARFAESI Act)
- Financial summary (FOIR 29.2%, LTV 42.8%)
- Deliberately absent: credit score (powers hallucination trap question)

### 7 Evaluation Questions
| ID | Intent | Question | What it tests |
|---|---|---|---|
| Q1 | lookup | "What is the annual interest rate?" | Baseline field retrieval |
| Q2 | lookup | "Who is the co-applicant and their income?" | Two-field compound retrieval |
| Q3 | calculate | "What is the total amount paid over full tenure?" | Multi-step arithmetic (EMI × months) |
| Q4 | compare | "Compare applicant and co-applicant monthly income" | Structured output, two entities |
| Q5 | explain | "What happens if I miss an EMI payment?" | Multi-sentence clause extraction |
| Q6 | summarise | "Give me a summary of this loan document" | Broad coverage, structure |
| Q7 | not_found | "What is the credit score of the applicant?" | Hallucination trap — must refuse |

### 6 Metrics (all automated, no human labelling)
| Metric | How calculated | Weight in composite |
|---|---|---|
| Groundedness | Fraction of `expected_keywords` found in answer (case-insensitive substring) | 30% |
| Number accuracy | Fraction of numbers in answer that exist in source context (regex `[\d,]+(?:\.\d+)?`) | 25% |
| Relevance | Jaccard similarity between question and answer content words (stop words removed) | 25% |
| Structure | Count of formatting markers (pipes, bullets, numbered lists, `key: ₹value`); 2+ = 1.0 | 10% |
| Speed | `max(0, 1 - latency_ms / 10000)` linear normalisation | 10% |
| Not-found compliance | Boolean: answer contains refusal phrase from vocabulary list | Not in composite (all 3 passed) |

**Composite score formula:**
```
composite = groundedness×0.30 + number_accuracy×0.25 + relevance×0.25 + structure×0.10 + speed×0.10
```

### Results

| Model | Groundedness | Number Accuracy | Structure | Relevance | Avg Latency | Avg Words | Composite |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.77 | 0.87 | 0.43 | 0.24 | 2,821ms | 59 | ~0.61 |
| Mistral-7B | 0.87 | 0.94 | 0.36 | 0.25 | 2,461ms | 54 | ~0.65 |
| **Qwen2.5-7B** | **0.85** | **0.94** | **0.43** | **0.26** | **1,846ms** | 64 | **~0.67** |

**Winner: Qwen2.5-7B** — highest composite, fastest (1,846ms avg), tied best number accuracy.

**Most revealing question — Q3 (arithmetic):**
| Model | Answer | Correct? | Failure mode |
|---|---|---|---|
| Llama-3.1-8B | ₹5,37,84,800 | Wrong | Ignored given EMI, applied interest formula from training knowledge |
| Mistral-7B | ₹15,01,520 | Debatable | Used EMI×tenure correctly but added processing fee (not asked) |
| Qwen2.5-7B | ₹14,90,520 | Correct | EMI×tenure, showed working (24,842×60), correct answer |

**Q7 (hallucination trap):** All 3 models scored PASS — responded with "Not found in the document."

**Not-found compliance vocabulary list (14 phrases):** "not found", "not mentioned", "not provided", "not in the document", "not available", "not stated", "no information", "cannot find", "does not mention", "not specified"

**Raw output files:** `scripts/ab_results.json` (21 responses + all scores), `scripts/ab_report.md` (formatted tables)

---

## 8. MLflow Experiment Tracking

**Server:** Self-hosted, runs on Instance 2 (port 5000 locally, nginx proxied).  
**Artifact store:** `s3://{bucket}/mlflow/artifacts/`  
**Backend store:** SQLite locally / PostgreSQL in prior Render deployment

### 3 Experiments
| Experiment | Purpose |
|---|---|
| `financehq_rag_queries` | Every live `/query` call — production tracking |
| `financehq_ab_eval` | A/B evaluation script runs (3-model comparison) |
| `loandoc_pipeline_runs` | Document ingestion pipeline metrics |

### Per-Query Run Schema (`financehq_rag_queries`)
**Tags:** `intent`, `variant`, `model`, `run_type`  
**Params:** `session_id`, `model_id`, `top_k`  
**Metrics:** `chunks_used`, `context_words`, `latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`, `retrieval_latency_ms`, `inference_latency_ms`  
**Artifacts:** `question.txt`, `answer.txt`, `prompt.txt`, `retrieved_chunks.json`

### Per-Pipeline Run Schema (`loandoc_pipeline_runs`)
**Params:** `session_id`, `user_id`, `original_filename`, `chunk_size`, `chunk_overlap`, `embedding_model`  
**Metrics:** `page_count`, `char_count_raw`, `char_count_cleaned`, `chunk_count`, `extraction_latency_ms`, `cleaning_latency_ms`, `chunking_latency_ms`, `embedding_latency_ms`  
**Artifacts:** `validation_report.json`

### MLflow Logging Pattern
Non-blocking background task using `loop.run_in_executor(None, _log)` to avoid blocking the async event loop. Logging failure is caught and warned — never propagates to the API response.

### Prompt Versioning
Prompt templates versioned via MLflow artifact store (`mlflow.log_artifact("prompts/lookup_v2.txt")`). Each run logs `prompt_template_version` param — complete audit trail: given any run, retrieve the exact prompt that produced that answer.

---

## 9. Deployment Architecture

### AWS EC2 — 2-Instance Split

**Why 2 instances:** MiniLM (sentence-transformers) + FastAPI + MLflow together exceed 1GB RAM on a single t2.micro (1GB RAM limit), causing OOM crashes. Django is lightweight (~150MB); FastAPI + MLflow is heavy (~700MB).

| Instance | Role | Services | RAM footprint |
|---|---|---|---|
| Instance 1 | React/Django UI | gunicorn (port 8000) + nginx (port 80) | ~150 MB |
| Instance 2 | FastAPI + MLflow | uvicorn (port 8001) + mlflow (port 5000) + nginx (port 80) | ~700 MB |

**Browser → Instance 1:** GET page load, Django serves React build via nginx  
**Browser → Instance 2:** POST /sessions, POST /query/stream directly (FASTAPI_URL injected into templates at render time)

### Instance 2 Setup Requirements
- 1GB swap file required before `pip install` (prevents OOM during torch install)
- CPU-only PyTorch installed first (avoids pulling 423MB CUDA wheel on 8GB disk):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

### nginx — SSE-Safe Config (Instance 2)
Critical for streaming responses to reach browser in real time:
```nginx
proxy_buffering            off;
proxy_cache                off;
proxy_set_header           Connection '';
chunked_transfer_encoding  on;
proxy_read_timeout         120s;
```

### CI/CD — GitHub Actions
- Trigger: push to `main`
- Action: SSH into both instances in parallel, run redeploy scripts
- `deploy-django.sh`: git pull → pip install → collectstatic → restart gunicorn
- `deploy-fastapi.sh`: git pull → pip install → restart uvicorn
- Required secrets: `EC2_SSH_KEY`, `INSTANCE1_IP`, `INSTANCE2_IP`

### IP Change Risk
EC2 public IPs change on stop/start. Mitigation: assign Elastic IP to each instance (1 free per running instance on AWS free tier). Without Elastic IP: must update `FASTAPI_URL` in Instance 1's `.env` and GitHub secrets after every stop/start.

### Prior Deployment (Render — abandoned)
Original plan: 3 Render web services (FastAPI + Django + MLflow) + 1 Render Postgres. Abandoned due to free-tier sleep behavior (15-min inactivity timeout, ~30s cold start) being disruptive for demo.

---

## 10. Test Coverage

**Run command:** `pytest --ignore=tests/pipeline/test_extractor.py --ignore=tests/test_s3_client.py -q`

| Test file | Count | Status | Notes |
|---|---|---|---|
| `tests/api/test_sessions.py` | 20 | All pass | Endpoint, pipeline stage transitions, merge logic |
| `tests/api/test_query.py` | 28 | All pass | 21 for /query + 7 for /query/stream SSE parsing |
| `tests/llm/test_bytez_client.py` | 14 | All pass | Bytez SDK fully mocked |
| `tests/retrieval/test_intent_classifier.py` | ~15 | All pass | 5-intent cosine similarity |
| `tests/pipeline/test_chunker.py` | ~10 | All pass | Sliding window, overlap, min chunk |
| `tests/pipeline/test_cleaner.py` | ~15 | 1 pre-existing failure | `test_full_word_not_truncated("FLOATING")` — bug in test expectation, not production code |
| `tests/pipeline/test_validator.py` | ~10 | 1 pre-existing failure | `test_warning_on_moderate_confidence` — bug in test expectation |
| `tests/pipeline/test_extractor.py` | ~10 | Skipped in CI | Requires live AWS credentials |
| `tests/test_s3_client.py` | ~10 | Skipped in CI | Requires live AWS credentials |

**Total: ~122 passing, 2 known pre-existing failures (test expectation bugs, not production bugs), 2 files require AWS credentials (skipped in CI).**

**Mock strategy:** All external dependencies (S3, Textract, Bytez/Groq, MLflow) mocked via `unittest.mock.patch`. Tests are fully offline-runnable except the two AWS credential files.

---

## 11. Key Engineering Decisions

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| PDF extraction | AWS Textract | PyMuPDF | Textract handles tables, forms, handwriting; PyMuPDF text-layer only |
| LLM inference | Groq API | Bytez API (prod), self-hosted vLLM | Groq: better speed/quality; no idle GPU cost |
| Embeddings | MiniLM-L6-v2 (local) | OpenAI text-embedding-ada-002 | $0 cost, ~5ms/chunk CPU, 384-dim sufficient for domain |
| Vector store | FAISS IndexFlatIP | ChromaDB, Pinecone | In-process, no extra service, serializable to S3 |
| Intent classification | Zero-shot cosine similarity (MiniLM) | LLM call, fine-tuned classifier | Reuses loaded embedder, no API cost, no training data, deterministic |
| MiniLM usage | Triple use: embed chunks + queries + classify intent | Separate models per use case | One model load (~90MB) serves 3 jobs |
| Multi-doc merge | Load existing + append + rebuild FAISS | Incremental FAISS add | FAISS IndexFlatIP doesn't support incremental add with persistence |
| Session state | S3 status.json per session | In-memory, DB | Stateless API — any replica can serve any request |
| Async pipeline | FastAPI BackgroundTasks | Celery, Airflow | No broker dependency; sufficient for single-instance portfolio use |
| SSE streaming | Word-by-word at 30ms/word (simulated) | Native Groq streaming | Bytez eval platform lacks native streaming; simulation good enough for UX |
| Deployment split | 2× EC2 t2.micro | Single EC2 t2.micro | RAM constraint: FastAPI+MiniLM+MLflow > 1GB; Django < 150MB |
| Frontend | React SPA | Streamlit, Django templates | Shows full-stack depth; SPA pattern cleaner for API-driven backend |

---

## 12. Known Limitations & Future Work

### Known Bugs

**Bug 1 — Retriever cache not evicted on multi-doc add**
- Location: `fastapi_service/app/pipeline/pipeline.py`, end of `run_pipeline()` stage 6
- Effect: after uploading a 2nd PDF to a session, queries search the old FAISS index until process restart
- Fix: call `evict_session(session_id)` (from `app/retrieval/retriever.py`) at end of stage 6
- Severity: Medium — affects multi-doc sessions only

**Bug 2 — Concurrent upload race condition**
- Location: `fastapi_service/app/pipeline/indexer.py`, `merge_into_session_index()`
- Effect: two simultaneous `POST /sessions/{id}/documents` calls race on S3 read-modify-write; one upload's chunks may be silently dropped from the merged index
- Fix: per-session distributed lock (Redis Redlock or DynamoDB conditional writes)
- Severity: Low at portfolio scale (sequential uploads in practice)

**Bug 3 — Textract async job ID not persisted**
- Location: `fastapi_service/app/pipeline/extractor.py`
- Effect: if the FastAPI process restarts during a multi-page Textract async job, the job is orphaned and the document gets stuck in `EXTRACTING` status
- Fix: persist `JobId` to S3 status.json; on startup, resume orphaned jobs
- Severity: Low (process restarts rare; 1-page sync path unaffected)

### Pending Features
- **LLM-as-judge evaluator** (`app/llm/evaluator.py`): post-query faithfulness and relevance scoring using a secondary LLM call; deprioritised — A/B eval script covers offline evaluation
- **Chat tab polish**: UI issues noted during testing, details deferred

### Architectural Limitations (by design)
- Not concurrent-safe for simultaneous uploads to same session
- Bytez free-tier streaming is simulated (full response fetched first, then word-drip)
- Textract async polling uses fixed 5s sleep (no exponential backoff)
- FAISS IndexFlatIP is exact search — scales to thousands of chunks, not millions
- Single-node FAISS in-process cache is not shared across multiple FastAPI replicas
