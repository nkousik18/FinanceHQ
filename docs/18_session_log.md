# Session Log

Dated record of working sessions: what changed, what was decided, and why. Newest first.
Add a new `## YYYY-MM-DD` section per session. Decisions get an ID (`S-<date>-<n>`) so later docs can cite them.

Related: initiative scope `loandoc-retrieval-evals-scope-v2.md` (its own Decision Log holds the D10 implementation choices) · R0 details `docs/r0_findings.md`.

---

## 2026-11-04 (cont.) — R1 step 1–2: data rules and EDGAR corpus

**Branch:** `feature/r1-corpus-parsing`

### Decisions

| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-13 | `SEC_USER_AGENT` uses a real but secondary email | SEC fair-access rules want a reachable contact; a fake one risks an IP block mid-download | `.env` (untracked) |
| S-2026-11-04-14 | **12 EDGAR deals approved** (list in scope v2 Decision Log) | Industry spread, smaller agreements, every deal has 2–3 amendments | `evals/corpus/edgar_selection.json` |
| S-2026-11-04-15 | Discovery, de-duplication, rendering and page-fidelity rules | See scope v2 Decision Log (2026-11-04 rows) | `loandoc-retrieval-evals-scope-v2.md` |

### Changes
- `.gitignore`: `data/` and evals PDFs (except `evals/fixtures/ci/`); **F9 test** `evals/tests/test_data_rules.py`.
- `evals/corpus/`: SEC client (rate-limited, User-Agent from env, on-disk cache), deal discovery, manifest, Chrome renderer with page fitting, fetch CLI.
- `evals/corpus/edgar_candidates.json` (32 checked candidates), `edgar_selection.json` (approved 12 + 4 reserves), `manifest.csv` (41 docs, checksums and page counts frozen).
- Corpus built locally: 41 PDFs, **1,836 pages** under `data/corpus/pdf/` (gitignored).
- 37 evals tests pass.

### Findings
- First-pass rendering inflated page counts (2,215 pages) because original pages spilled over; fixed by per-document zoom (now 1,836).
- `deal_09_ca` (United Therapeutics): Chrome ignores some of its 157 page breaks, so it renders as 108 pages; text is intact (97%). PDF page numbers, not original page numbers, are the reference for evidence labels.
- 56 near-empty pages across the corpus, mostly signature pages at the end of agreements.
- Three agreements (Raymond James, Harmonic, TrueBlue) were filed as page images plus text; the text layer is complete.

### Open items
- Next in R1: CFPB consumer forms, scan simulator, parser interface + Docling, ingestion labeling, §8.1 metrics, S0.

---

## 2026-11-04 — Retrieval & Evals initiative: scope v2 and R0

**Branch:** `feature/r0-baseline-fixes` → merged to `main`

### Context
- The AWS free tier ended. The backend on AWS (2× EC2, S3 bucket, Textract) was deleted to stop charges. The live demo is down.
- A scope for a "Retrieval & Evals" initiative (`loandoc-retrieval-evals-scope.md`, v1) was written on 2026-11-01/03. It assumed AWS and ~60 private loan PDFs; the repo only has `loan1.pdf`.

### Decisions

| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-1 | Replace Textract with **Docling** (local, open source) for the evals, behind a parser interface | No AWS; $0 per page; a later parser comparison becomes a config change | Scope v2 §0, §6.0, D4 |
| S-2026-11-04-2 | Use a **public corpus**: SEC EDGAR credit agreements with their amendments, CFPB loan forms, simulated scans | Reproducible and showable; amendments give hard, realistic cross-document questions | Scope v2 §5.0 |
| S-2026-11-04-3 | Write a **new scope (v2)** instead of editing v1; keep every v1 metric and evaluation | The initiative's purpose is the evaluation catalog; v1 kept for history | `loandoc-retrieval-evals-scope-v2.md` |
| S-2026-11-04-4 | Private-data rules relaxed: golden sets, labels and examples are committed; raw PDFs fetched by script, not committed | Corpus is public; PDFs are large | Scope v2 D11, F9–F11 |
| S-2026-11-04-5 | Everything runs **locally**: local storage, local MLflow; no cloud except the Groq API | No AWS | Scope v2 D8 |
| S-2026-11-04-6 | Complex/multi-column layout pages may come from **consumer forms** | EDGAR agreements are mostly single-column | Scope v2 §15 Q2 |
| S-2026-11-04-7 | Groq is on the **free tier: 200,000 tokens/day** | User's account | Scope v2 §15 Q3 |
| S-2026-11-04-8 | **PII module and Airflow DAG** built as placeholders, **not connected** to the pipeline or evals | Wanted in the project; modelled on `loan-doc-ai-end-to-end` (guardrails) and `LoanQA-MLOps` (OCR DAG) | `fastapi_service/app/privacy/`, `orchestration/` |
| S-2026-11-04-9 | Generator **`qwen/qwen3.8-27b`**, judge **`openai/gpt-oss-120b`** | `llama-3.3-70b-versatile` was retired by Groq (404); judge must be a different family than the generator | Scope v2 D4; `app/core/config.py` |
| S-2026-11-04-10 | Load test (S5) only on the **baseline and the S4 winner**: 150 requests each, concurrency 1, p50/p95 only | Original plan (~8M tokens, ~40 days) can't fit 200k tokens/day; cut plan is ~0.6M tokens, ~3 days | Scope v2 §9.4 |
| S-2026-11-04-11 | Disable the EC2 auto-deploy workflow (manual trigger only) | Hosts no longer exist; every merge to `main` would fail | `.github/workflows/deploy.yml` |
| S-2026-11-04-12 | Skip (not delete) the two stale test modules, with reasons | They test removed code (Bytez SDK client, old Textract helpers); rewriting is tracked debt | `docs/r0_findings.md` §5 |

### Changes (R0)
- **Bug fix:** retriever cache is evicted after a document is indexed (`app/pipeline/pipeline.py`) + regression test.
- **Bug fix:** `_is_truncated` in the cleaner flagged every value ending in a letter; now only cut-off URLs.
- **Tests repaired:** stream tests mocked offline (they were calling live Groq); validator fixture fixed; `moto` + `setup_method` fix for S3 tests; `requirements-dev.txt` added. Result: 169 passed, 2 skipped (was: 3 import errors, 6 failures).
- **Measured:** MiniLM truncation on `loan1.pdf` — 100% of C1 chunks exceed the 256-token limit; 22.5% of the document's words are never embedded (`scripts/r0_truncation_check.py`, `docs/r0_truncation.json`).
- **Model switch:** `qwen/qwen3.8-27b` in `app/core/config.py`, `.env` (local, untracked), `deploy/*.example`. Verified with a live streaming call.
- **Placeholders:** `app/privacy/pii.py` (26 tests); `orchestration/` Airflow DAG + stub tasks (6 tests; DAG syntax-checked only, Airflow not installed).
- **Docs:** scope v1 and v2 added at repo root; `docs/r0_findings.md`; this log; status notes in `docs/16_build_state.md` and `docs/14_roadmap.md`.

### Open items
- R1 next: EDGAR deal candidates for approval (12 deals), corpus fetch script, Docling parser, ingestion labeling and metrics, S0.
- Needs from Kousik before R1 downloads: `SEC_USER_AGENT="Name email"` in `.env`; approval of the deal list.
- Scope v2 Open Question 1 (real scanned documents) still open.
- Stale docs still name the old Llama model: `docs/17_deployment_aws.md`, `docs/LoanDoc_resume_project_doc.md`, `financehq.md`, React architecture page.
- Known, unfixed: concurrent-upload race on the session index; Textract async job ID not persisted (both production-only, AWS-bound).
