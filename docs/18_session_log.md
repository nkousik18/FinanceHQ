# Session Log

Dated record of working sessions: what changed, what was decided, and why. Newest first.
Add a new `## YYYY-MM-DD` section per session. Decisions get an ID (`S-<date>-<n>`) so later docs can cite them.

Related: initiative scope `loandoc-retrieval-evals-scope-v2.md` (its own Decision Log holds the D10 implementation choices) · R0 details `docs/r0_findings.md`.

---

## 2026-11-06 — OCR decision and agent handoff

**Branch:** `feature/r1-corpus-parsing`

| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-06-1 | **Apple Vision OCR text downstream; Tesseract for confidence metrics and H7** | Vision reads heavy scans far better (numbers 73% vs 41%); Tesseract's confidence is the meaningful one | Scope v2 Decision Log; `evals/parsers/config.py` (`downstream_config`) |
| S-2026-11-06-2 | Hand off to a fresh agent; handoff brief in `docs/19_handoff.md` | Conversation context was getting large | `docs/19_handoff.md` |
| S-2026-11-06-3 | `scan_01_p001` table t3: deleted stray cell (6,1) `"a"` that overlapped merged cell 20 | OCR junk inside a 3-row merged cell; Kousik asked for the fix, then verified the page (30/30) | label file (Kousik's) |
| S-2026-11-06-4 | **S0 pre-registration** written before the run: `docs/20_s0_results.md` (H1–H10 verbatim, S0 tests H6 + H10, decision/H6/H10 rules) | Scope §11 rule | `docs/20_s0_results.md`; scope Decision Log |
| S-2026-11-06-5 | **H6 disclosed as not cleanly pre-registered**: Auto-only, outcome already seen in dry runs, pairing rule set after a first version | Honesty; reported with the result | `docs/20_s0_results.md` §2 |
| S-2026-11-06-6 | **H10 test** = decision's paired bootstrap restricted to `table_heavy` tables; CI > 0 supported, < 0 not, else inconclusive. `run_s0.h10()` + 4 known-answer tests | H10 had no computation in the runner | `evals/ingest_metrics/run_s0.py`, `evals/tests/test_run_s0.py` |

Later: 132 evals tests pass. Golden set 30/30 verified.

Changes: `P2V` config added (accurate tables + Apple Vision), parsed on the 40 scans (103 s, 0 failures); `downstream_config()` + tests (128 evals tests pass).

State: Kousik has verified 29 of 30 golden-set pages in the labelling app (files not yet committed by Kousik).

### Open items
- Kousik: verify the last golden page; commit the label files.
- Next agent: run S0 on verified labels, write the S0 results doc, finish R1 (see `docs/19_handoff.md`).

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

### CFPB consumer forms (later the same day)
| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-16 | Use all 11 completed English CFPB TRID samples (5 Loan Estimates, 6 Closing Disclosures) | Dense fee tables and form layouts the credit agreements lack; public domain | `evals/corpus/cfpb.py` |
| S-2026-11-04-17 | Kousik downloads the PDFs by hand; the script imports and checksums them | consumerfinance.gov blocks scripts (403); no bot-protection workarounds | Scope v2 Decision Log |
| S-2026-11-04-18 | 7 sessions, grouped by matching borrower/property/loan amount | Avoids contradictory figures inside one session | Scope v2 Decision Log |

Result: corpus = 41 EDGAR docs (1,836 pages) + 11 CFPB forms (40 pages). 39 evals tests pass.

### Simulated scans (later the same day)
| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-19 | 40 simulated-scan pages at 3 degradation levels, with fixed quotas per stratum | Exact reference text for scanned-page accuracy (H6); real scans can't give that | Scope v2 Decision Log; `evals/corpus/scan_specs.json` |
| S-2026-11-04-20 | Exclude signature pages; cap pages per document per stratum | Signature pages say little about reading accuracy; tables cluster in few documents | Scope v2 Decision Log |

Result: corpus = 41 EDGAR docs (1,836 pages) + 11 CFPB forms (40 pages) + 40 simulated-scan pages (from 24 source docs). Builds are reproducible (byte-identical images). 46 evals tests pass.

### Docling parsing and the labeling tool (later the same day)
| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-21 | Run Docling on the Apple GPU (MPS) | Kousik's request; measured ~10% faster than CPU | Scope v2 Decision Log |
| S-2026-11-04-22 | Docling in a separate venv; Tesseract OCR; parse cache as the interface | Protects the app venv; Tesseract is the only engine with both usable accuracy and meaningful confidence | Scope v2 Decision Log |
| S-2026-11-04-23 | Exclude CFPB cover sheets from scan and golden-set selection | They're covers, not forms | Scope v2 Decision Log |
| S-2026-11-04-24 | Golden-set selection rules and label conventions | See Decision Log | `evals/golden/ingest/` |

Results: full parse of 92 docs × 2 configs in ~40 minutes, 0 failures. 30 labeling pages selected and pre-filled (25 from Docling P1, 5 empty for from-scratch labeling), images committed. 62 evals + 169 app tests pass.

Findings so far: Docling found no key-value pairs (flat PDFs); it missed a table PyMuPDF detected (`deal_02_ca_p021`); one Docling table on `scan_01` has overlapping cells.

### Labelling app (later the same day)
| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-04-25 | Build a local click-and-drag labelling app instead of hand-editing JSON | Kousik found the JSON/coordinate workflow unclear; the app should cut labelling time | Scope v2 Decision Log; `evals/golden/ingest/app/` |

Also fixed: overlay images drew boxes in the wrong place on Chrome-rendered pages (inherited scaling transform).

### S0 metrics drafted (later the same day)
§8.1 metrics implemented in `evals/ingest_metrics/` (normalize, text, geometry, tables, layout, confidence), `evals/stats.py` (bootstrap, paired diff, Holm), S0 runner `run_s0.py` (golden + full corpus, decision rule, optional MLflow). 126 evals tests pass. Dry run (unverified labels, so golden numbers are not results) runs end to end in ~35 s.

Corpus findings (Auto metrics; real, not dependent on labels):
- SEC agreements: 38 of 41 documents read almost perfectly (numeric fidelity 99.9%, median CER 0.1%). The 3 filed as page images + text (Raymond James, Harmonic, TrueBlue) lose doubled letters ("11.02" → "1 .02", "Borrower" → "Bor ower"): numeric fidelity 80.4%.
- Docling drops some page-footer text entirely (e.g. "77 CHAR1\1635542v6").
- CER against the text layer is dominated by reading order on CFPB forms (31% CER, 99.7% order-free word recall).
- OCR on heavy scans: Tesseract numbers right 41%, Apple Vision 73%; Tesseract confidence is informative, Vision's is not (mean 0.99).

Other changes and notes from this block:
- New dependencies (app venv, `requirements-dev.txt`): `rapidfuzz` 3.14.6 (fast edit distance for CER/WER), `apted` 1.0.3 (tree edit distance for TEDS). Installed without changing any existing package.
- The pre-fill and the metrics now share one Docling reader (`evals/parsers/docling_doc.py`). Checked: on the 23 pre-filled pages not yet edited, it reproduces the committed labels exactly.
- Incident: end-to-end testing of the labelling app edited one real label file (`cfpb_le_h24b_p003`); restored from git right after. No labels were lost.
- Kousik started labelling in the app (7 files edited so far). Those files are deliberately left out of agent commits until Kousik commits them.

### Open items
- **Decision needed (Kousik):** OCR engine for the text used downstream (Tesseract vs Apple Vision); see the trade-off above.
- **Kousik:** label the 30 pages in the app (`venv/bin/python -m evals.golden.ingest.app.server`, open http://127.0.0.1:8765), from-scratch pages first; guide in `evals/golden/ingest/README.md`.
- Next in R1 (agent, can proceed in parallel): §8.1 metric implementations with F8 tests; then S0 once labels are verified.

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
