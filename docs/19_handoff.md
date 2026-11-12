# Agent handoff — Retrieval & Evals initiative (2026-11-06)

For a fresh agent picking up FinanceHQ / LoanDoc. Read this first, then the files in "Read next".

## 1. Who you're working with and how

- **Kousik** (owner). Portfolio project for AI-engineering job applications.
- **Style:** short, direct answers ("caveman": no filler). Plain words; explain jargon when you first use it.
- **Work step by step.** One module at a time, with tests. **Ask before moving to the next milestone/phase.**
- **Git:** never commit to `main`. Feature branches (`feature/<name>`), merged by PR (`gh pr create` / `gh pr merge --merge`).
  After each chunk of work, commit + push (or remind Kousik with the exact command) — he usually says "commit and push".
  Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies end with the Claude Code line.
- **Log everything.** Every decision/finding/change goes into `docs/18_session_log.md` (new `## YYYY-MM-DD` section, IDs `S-<date>-<n>`)
  and implementation choices into the **Decision Log** at the end of `loandoc-retrieval-evals-scope-v2.md`. Kousik checks this.
- **Don't commit Kousik's label files** (`evals/golden/ingest/pages/*.json`) unless he asks — stage your own files explicitly, never `git add -A`.
- If a fixed decision (scope §14) looks wrong in practice: stop and ask, don't work around it.

## 2. Read next (in this order)

1. `loandoc-retrieval-evals-scope-v2.md` — the plan: metrics (§8), stages S0–S5 (§9.3), requirements (§10), hypotheses (§11), milestones R0–R8 (§12), decisions (§14), **Decision Log** (end).
2. `docs/18_session_log.md` — what happened each session, newest first.
3. `docs/r0_findings.md` — R0 measurements (MiniLM truncation, test repair, model switch).
4. `evals/golden/ingest/README.md` — golden-set labelling guide.

## 3. Where things stand

| Milestone | Status |
|---|---|
| R0 | ✅ Merged to `main` (PR #2): cache-eviction fix, tests repaired, truncation measured (100% of C1 chunks > MiniLM 256 tokens), Groq model switch, placeholder PII module + Airflow DAG |
| R1 | 🔶 In progress on branch **`feature/r1-corpus-parsing`** (pushed) |

**R1 done:**
- Public corpus, all local, no AWS: 41 SEC EDGAR docs (12 credit-agreement "deals" + amendments, 1,836 pages), 11 CFPB TRID sample forms (40 pages), 40 simulated scans. Manifest with checksums: `evals/corpus/manifest.csv`.
- Docling 2.147.0 parsed everything: configs **P1** (fast tables) / **P2** (accurate tables), Tesseract OCR, Apple GPU (MPS). Plus **P1V/P2V** (Apple Vision OCR) on the 40 scans.
- Ingestion golden set: 30 pages selected and pre-filled; local labelling app built; **Kousik has verified 29/30** (his files are uncommitted).
- §8.1 metrics + S0 runner written and tested (128 evals tests); dry run works.

**R1 remaining (in order):**
1. ~~Golden set verified and committed~~ (30/30). ~~Pre-registration~~ (`docs/20_s0_results.md` §1–3).
2. ~~S0 run~~ (2026-11-12): **P2 wins**; H10 supported; H6 not supported. Disputed tables were settled by the agent at Kousik's request (S-2026-11-12-1, disclosed in §4.0). Re-render tables with `venv/bin/python -m evals.ingest_metrics.s0_report`.
3. Downstream config from now on: `downstream_config("P2", part)` (P2 digital, P2V scans).
4. Not yet implemented from §8.1: cleaning metrics (repair counts, chars removed, over-cleaning rate), per-stage latency beyond parsing, validation warnings/errors per doc. They need an adapter from Docling's output into the production `CleanedResult`/cleaner/validator — the same adapter R2 needs. Split-table rate is a recorded gap (per-page labels can't show it).
5. Open the R1 PR to `main`; ask Kousik before starting R2.

**After R1:** R2 (chunker interface with provenance; C1 = exact production chunker, N2 equality test; C2–C5), R3 (QA golden set, 120 questions), R4 (retrieval metrics, S1), then **stop for review** (D9).

## 4. Decisions you must respect (most recent)

- **Downstream text:** simulated scans use Apple Vision OCR (`P1V`/`P2V`); digital docs use P1/P2. Use `evals/parsers/config.downstream_config(table_config, part)`. Confidence metrics and H7 use the Tesseract run (P1/P2). (S-2026-11-06-1)
- **LLMs (Groq free tier, 200k tokens/day):** generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b` (counts hidden reasoning tokens). S5 load test = baseline + winner only, 150 req each, concurrency 1, p50/p95. Spend caps D7.
- **No AWS anywhere.** Production app still imports S3/Textract and can't run; porting it is out of scope (R8 recommends).
- Data rules D11: corpus is public; raw PDFs/caches under `data/` are gitignored and rebuilt by scripts.

## 5. Environment and commands

| What | How |
|---|---|
| App + evals venv | `venv/` (Python 3.12). Tests: `venv/bin/python -m pytest -q evals/tests` and `cd fastapi_service && ../venv/bin/python -m pytest -q` (169 pass, 2 stale skipped) |
| Docling venv | **`venv-docling/`** (separate: Docling's numpy 2/torch/httpx conflict with the app venv). Pinned in `requirements-docling.txt` |
| Rebuild corpus | `venv/bin/python -m evals.corpus.fetch fetch` (needs `SEC_USER_AGENT` in `.env`; CFPB PDFs must be in `data/corpus/inbox/cfpb/` — consumerfinance.gov blocks scripts) ; scans: `python -m evals.corpus.scans build` |
| Parse | `caffeinate -i venv-docling/bin/python -m evals.parsers.docling_runner --configs P1 P2` (resumable; ~40 min for both on the M3) |
| Labelling app | `venv/bin/python -m evals.golden.ingest.app.server` → http://127.0.0.1:8765 ; CLI: `python -m evals.golden.ingest.labeling status|validate|render` |
| S0 | `venv/bin/python -m evals.ingest_metrics.run_s0 [--include-unverified] [--skip-corpus] [--mlflow]` → `data/results/s0/` ; MLflow store `data/mlflow/mlflow.db` |

Tools on the Mac: Chrome 156 (HTML→PDF rendering), Tesseract 5.5.1, Apple M3 GPU (MPS), 16 GB RAM.

## 6. Code map

```
evals/corpus/      sec_client, edgar_discover, manifest(.csv), render (Chrome), fetch, cfpb, scans, *_candidates/selection/scan_specs.json
evals/parsers/     config (P1/P2/P1V/P2V, cache keys, downstream_config), docling_runner (venv-docling), docling_doc (cache → per-page views)
evals/golden/ingest/  schema (labels + F4 validator), labeling (select/prefill/render), app/ (server + index.html), pages/, images/, selection.json
evals/ingest_metrics/ normalize, text, geometry, tables (incl. TEDS), layout, confidence, run_s0
evals/stats.py     bootstrap, ratio CI, paired diff CI, Holm
evals/tests/       all evals tests (F8 known-answer tests per metric; F9 data-rules test)
orchestration/     placeholder Airflow DAG (not wired in)   fastapi_service/app/privacy/pii.py  placeholder PII (not wired in)
```

## 7. Findings so far (for the S0 report)

- Docling reads 38/41 SEC docs almost perfectly (numeric fidelity 99.9%). The 3 filed as page images + text (deal_03 Raymond James, deal_07 Harmonic, deal_11 TrueBlue) **lose doubled letters** ("11.02" → "1 .02"): numeric fidelity 80.4%.
- Docling drops some page-footer text entirely; finds **no key-value pairs** on flat CFPB forms (Kousik labels them by hand); missed a table on `deal_02_ca_p021`.
- CER vs the text layer is dominated by **reading order** on forms (CFPB: CER 31%, order-free word recall 99.7%) → always report `bag_word_recall` next to CER.
- Heavy scans: Tesseract numbers right 41% / CER 81%; Apple Vision 73% / 5.5%. Vision confidence ≈ 0.99 always (uninformative).
- H6 as specified (scan CER ≥ 2× its source page's CER) currently **not supported** because digital-page CER is itself order-inflated — explain with the word-recall diagnostic.

## 8. Gotchas learned the hard way

- **zsh doesn't word-split `$VAR`** — use `${=VAR}` when passing lists (e.g. `--only ${=SCANS}`).
- Chrome-rendered PDFs leave a scaling transform active: call `page.wrap_contents()` before drawing on them with PyMuPDF.
- Docling boxes come in two origins (BOTTOMLEFT for items, TOPLEFT for table cells); `docling_doc.to_topleft` handles it. List numbers live in `marker`, not `text`.
- Every CFPB sample PDF starts with a cover sheet (`evals/corpus/cfpb.is_cover_page`).
- EDGAR: same amendment is often filed twice (8-K, then 10-Q; redacted then full) — dedupe logic in `edgar_discover.dedupe_amendments`.
- Unverified (pre-filled) golden labels **are** Docling P1 output — any run on them is a dry run, not a result.
- Background parse runs: launch with `run_in_background`; don't poll; the runner logs to `data/parse_cache/run.log`.
