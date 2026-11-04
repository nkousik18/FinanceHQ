# R0 Findings — Retrieval & Evals Initiative

**Date:** 2026-11-04 · **Branch:** `feature/r0-baseline-fixes` · **Scope:** `loandoc-retrieval-evals-scope-v2.md` §12 R0

## 1. Retriever cache eviction bug — fixed

`run_pipeline()` merged a new document into the session's S3 index but never called `evict_session()`, so queries kept using the stale in-memory index until restart.

- Fix: `evict_session(session_id)` right after `merge_into_session_index()` (`fastapi_service/app/pipeline/pipeline.py`). Not called when indexing fails, so the still-current index stays cached.
- Regression test: `fastapi_service/tests/pipeline/test_pipeline_cache_eviction.py` (4 tests). 2 of them fail on the pre-fix code and pass after.

## 2. MiniLM truncation (H8 baseline)

Script: `scripts/r0_truncation_check.py` · Output: `docs/r0_truncation.json`
Input: the saved Textract response for `loan1.pdf` (`scripts/output/loan1_raw.json`, 7 pages), run through the production path (`_assemble_result` → `clean_extraction` → `chunk_document` with C1 defaults). No AWS call.

| Measure | Value |
|---|---|
| Embedder `max_seq_length` (confirmed from the loaded model) | **256** tokens (254 after [CLS]/[SEP]) |
| C1 chunks | 16 |
| MiniLM tokens per chunk (p5 / p50 / p95 / max) | 365 / 426 / 565 / 600 |
| MiniLM tokens per word | 1.47 (the chunker assumes 1.33) |
| **Chunks over the limit (truncation rate)** | **100%** |
| Tokens never embedded, per chunk copy | 43.0% |
| **Document words no chunk embeds at all** (overlap de-duplicated) | **22.5%** |

**What it means:** every production chunk is cut off by the embedder. Each chunk is embedded from roughly its first 170 words; the rest is still sent to the LLM as context but was never searchable. About a fifth of the document can't be found by dense retrieval at all. Overlap rescues only ~40 words per boundary.

**Caveats:** one 7-page document (Indian loan application, Textract output). R2 re-measures on the v2 corpus with Docling output. This supports H8's first half (C1 > 50%) on this document only; it is not the H8 result.

## 3. Groq model retired — resolved

`llama-3.3-70b-versatile` returns `404 model_not_found` for the configured key. Models the key can use (2026-11-04): `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, plus guard/speech/Arabic models.

**Decision (Kousik, 2026-11-04):** generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`.
- Set in `app/core/config.py` (was a leftover `c4ai-aya-expanse-32b`), local `.env`, and both `deploy/*.example` files.
- Live checks: both models answered a test prompt correctly; the app's `stream_tokens` streamed a clean Qwen answer (no reasoning text in the content).
- gpt-oss-120b returns hidden reasoning tokens (88 billed output tokens for a 2-word reply). Judge budgets must count them.
- Not updated (historical or out of scope): `docs/17_deployment_aws.md`, `docs/LoanDoc_resume_project_doc.md`, `financehq.md`, and the React architecture page (`react_ui/src/components/architecture/steps.ts`; UI changes are a non-goal).

## 4. Groq free tier: 200,000 tokens/day — S4 and S5 don't fit as written

Rough per-call size: ~1,600 input tokens (1,200-word context) + ~200 template + ~200 output ≈ **2,000 tokens**. Judge calls ≈ 2,500.

| Stage | Calls | Tokens | Days at 200k/day |
|---|---|---|---|
| Judge calibration (§8.6) | 50–75 gen + judge | ~0.3M | ~1.5 |
| S4 generation (80 × 4 configs) | 320 | ~0.64M | ~3.2 |
| S4 judging | 320 | ~0.8M | ~4 |
| Noise sensitivity + consistency (dev) | ~160 | ~0.3M | ~1.5 |
| **S5 load test as written** (500 × 4 configs × 2 passes) | 4,000 | **~8M** | **~40** |

If the daily limit is per model (common on Groq), generator and judge budgets are separate and S4 shrinks to ~4 days.

**Decision (2026-11-04):** S5 load-tests only the baseline and the S4 winner: 150 requests each at concurrency 1, p50/p95 only, ~0.6M tokens over ~3 days (scope v2 §9.4).

## 5. Test suite repair

The recorded baseline ("~122 passing, 2 failing") was out of date. Actual state on `main`: 3 test files failed to import, 6 tests failed, and 4 stream tests were calling the live Groq API.

| Problem | Fix |
|---|---|
| Stream tests mocked `get_llm_client().complete`, but `/query/stream` now uses native `stream_tokens`, so the tests hit live Groq | Mock `stream_tokens` with an offline async generator. The LLM-failure test now expects the real behavior: HTTP 200 with a final `{"error": ...}` SSE event (the stream has started, so a 502 is impossible) |
| `_is_truncated` flagged any value ending in a letter (`r"[a-zA-Z]$"`), e.g. "FLOATING" | **Code bug, fixed:** only URLs whose last dot-segment is one letter. 3 edge-case tests added. Only affects the `is_truncated` flag on form fields |
| Validator test expected a pass but its fixture had no form fields or tables (`NO_STRUCTURED_DATA` is an error) | Fixture now includes one form field |
| `test_s3_client.py`: `moto` not installed; `setup_method` signature broke under pytest 9 + `@mock_aws` | Added `requirements-dev.txt` (pytest, pytest-asyncio, moto); `setup_method(self, method=None)` |
| `test_bytez_client.py` targets the removed Bytez SDK client | **Skipped with reason.** Debt: write `GroqClient` tests |
| `test_extractor.py` targets removed extractor helpers (`_parse_blocks`) | **Skipped with reason.** Textract is out of evals scope. Debt: rewrite against `_parse_text_blocks` / `_assemble_result` if Textract stays in production |

Result: **143 passed, 2 skipped** in `fastapi_service/tests`; plus 26 PII and 6 orchestration tests below.

## 6. Other R0 changes

- `.github/workflows/deploy.yml`: auto-deploy on push to `main` disabled (manual `workflow_dispatch` only); the EC2 hosts no longer exist.
- **Placeholder PII module** `fastapi_service/app/privacy/pii.py` + 26 tests. Detects SSN, EIN, card (Luhn), labelled routing (ABA checksum) and account numbers, email, phone, PAN; mask / hash / partial redaction. Patterns adapted from `LoanGuardrails` in `loan-doc-ai-end-to-end`. **Not imported anywhere in the app.**
- **Placeholder Airflow DAG** `orchestration/dags/loandoc_ingestion_dag.py` + stub tasks `orchestration/tasks/ingestion_tasks.py` + 6 tests. Layout adapted from `LoanQA-MLOps/dags/ocr_pipeline_dag.py`. **Not connected to anything.** Airflow is not installed, so the DAG file is syntax-checked only, not run under Airflow.
