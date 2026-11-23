# Agent handoff — Retrieval & Evals initiative (updated 2026-11-22)

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
   `docs/20_s0_results.md` (S0 parsing), `docs/21_r2_chunk_metrics.md` (R2 chunk metrics) and
   **`docs/22_s1_retrieval.md` (S1 retrieval)** — generated results (F7: never hand-edit numbers).
4. `evals/golden/ingest/README.md` — golden-set labelling guide.

## 3. Where things stand

| Milestone | Status |
|---|---|
| R0 | ✅ Merged (PR #2): cache-eviction fix, tests repaired, MiniLM truncation measured, Groq model switch, placeholder PII + Airflow DAG |
| R1 | ✅ Merged (PR #3, #4): public corpus, Docling parser, ingestion golden set, S0 → **P2 wins** |
| R2 | ✅ Merged: chunkers C1–C5 (`evals/chunking/`) + §8.2 chunk metrics (`docs/21_r2_chunk_metrics.md`) |
| R3 | ✅ Merged (PR #7): QA golden set `evals/golden/qa/questions.jsonl`, q0001–q0120, **40 dev / 80 test**, all human-verified |
| R4 | ✅ Merged (PR #8): retrieval metrics, runner, MLflow, report; S1 → **C2 + C4** |
| R5 | 🔶 On **`feature/r5-retrieval-strategies`**. Steps (S-2026-11-22-31): ~~0 housekeeping~~ → ~~1 `R-bm25` + `R-hybrid`~~ → ~~2 `R-pc`~~ → **3 `+rerank`** → 4 S2 run + report → 5 S3 → PR. Ask before each step |

**R4 (merged), what exists:**
- `evals/retrieval/metrics.py` — all §8.3 metrics per question. Coverage = R2's `covers()` rule. Budget = copy of production `router._build_context` (greedy to 1,200 words, first chunk always in); a test pins `MAX_CONTEXT_WORDS` to production.
- `evals/retrieval/index_cache.py` — chunk/embedding cache (`data/cache/chunks/`), key = chunker id + params + parser cache key + cleaning version + embedder + dropped pages.
- `evals/retrieval/views.py` — per-question index view: digital session docs; a question citing a **scan** gets that scan added and its source page dropped (a global swap would hide pages 15 digital questions cite).
- `evals/retrievers/dense.py` — `R-dense` (FAISS IndexFlatIP, normalized MiniLM, no threshold = production). Registry `RETRIEVERS` lives in `evals/retrievers/__init__.py`; interface `search(index, query_vec, k, query=...)` + `needs_vector`.
- `evals/retrievers/bm25.py` (R5) — `R-bm25`: bm25s, own tokenizer (no stemming, numbers normalized), only score > 0 returned. `hybrid.py` — `R-hybrid`: RRF k=60 over dense + BM25 lists max(20, k) deep. `parent_child.py` — `R-pc`: C2 children → C4 sections rebuilt from page text, 800-token window cap, child skipped if ≥ 50% inside a returned parent; run config needs `parent_chunker: C4`.
- `evals/run.py` — runner; YAML configs in `evals/configs/` (explicit run list, never a generated product). Ranks max(top_k, 10); only top_k can enter the budget. Writes `data/results/r4/<stage>/<split>/<run>/{questions.jsonl,summary.json}` and logs one MLflow run per config.
- `evals/tracking.py` — local MLflow (`data/mlflow/mlflow.db`), experiments `loandoc_retrieval_evals` (S1–S4), `loandoc_ingestion_evals` (S0). `@`→`_at_`, `type=x`→`type.x` in metric names.
- `evals/report.py` — `--stage S1 --split dev` → `docs/22_s1_retrieval.md`: ★ metrics with question-bootstrap CIs, paired comparisons vs baseline (C1) and between rank neighbours, tie detection, generated notes. Refuses smoke/mixed runs. `PICKS = {"S1": 2, "S2": 3}`; `DOCS` maps (stage, split) → doc (add S2 there).

**S1 result (dev, n = 33 answerable, top_k 20):** recall@budget C2 **0.773**, C4 0.758, C5 0.758, C3 0.652, C1 0.515. C2/C4/C5 beat C1 (CIs exclude 0), indistinguishable from each other. C4 = C5 tie → **faster wins** (Kousik) → **S1 pick: C2 + C4**. Evidence fragmentation = 0 for all chunkers (spans ≈ one sentence; checked independently).

**R5 plan (approved 2026-11-22):** R5 = `R-bm25`, `R-hybrid` (dense top-20 + BM25 top-20, RRF k=60), `R-pc` (parent–child), `+rerank` (cross-encoder/ms-marco-MiniLM-L-6-v2, top 20 → k); **S2** = C2, C4 × {dense, bm25, hybrid, pc} ± rerank (≤ 16 runs, dev, top 3 + C1-dense baseline by recall@budget); **S3** = winner × {Flat, HNSW, IVF}, recall vs exact + latency, no pick. Kousik's choices: **one R-pc config (C2 children → C4 section parents, 800-token window cap) → S2 = 14 runs**; S3 = per-question views + one pooled whole-dev index. Baseline C1-dense comes from the S1 run (hash-checked).

## 4. Decisions you must respect (most recent)

- **Downstream text:** simulated scans use Apple Vision OCR (`P1V`/`P2V`); digital docs use P1/P2. Use `evals/parsers/config.downstream_config(table_config, part)`. Confidence metrics and H7 use the Tesseract run (P1/P2). (S-2026-11-06-1)
- **LLMs (Groq free tier, 200k tokens/day):** generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b` (counts hidden reasoning tokens). S5 load test = baseline + winner only, 150 req each, concurrency 1, p50/p95. Spend caps D7.
- **No AWS anywhere.** Production app still imports S3/Textract and can't run; porting it is out of scope (R8 recommends).
- Data rules D11: corpus is public; raw PDFs/caches under `data/` are gitignored and rebuilt by scripts.
- **QA set is frozen.** A fix = new qid + `retired: true` on the old one (F3). Never renumber or edit in place.
- **S1–S3 pick on dev only; test is for S4.** One decision metric per stage (D3): recall@budget for S1/S2. Ties → faster.
- **S1 `top_k` = 20** (the 1,200-word budget binds, not chunk count). Keep it for S2 unless Kousik says otherwise.
- Every reported number comes from a generated report (F7).

## 5. Environment and commands

| What | How |
|---|---|
| App + evals venv | `venv/` (Python 3.12). Tests: `venv/bin/python -m pytest -q evals/tests` and `cd fastapi_service && ../venv/bin/python -m pytest -q` (evals **398** pass; app **173** pass, 2 stale skipped) |
| Docling venv | **`venv-docling/`** (separate: Docling's numpy 2/torch/httpx conflict with the app venv). Pinned in `requirements-docling.txt` |
| Rebuild corpus | `venv/bin/python -m evals.corpus.fetch fetch` (needs `SEC_USER_AGENT` in `.env`; CFPB PDFs must be in `data/corpus/inbox/cfpb/` — consumerfinance.gov blocks scripts) ; scans: `python -m evals.corpus.scans build` |
| Parse | `caffeinate -i venv-docling/bin/python -m evals.parsers.docling_runner --configs P1 P2` (resumable; ~40 min for both on the M3) |
| Labelling app | `venv/bin/python -m evals.golden.ingest.app.server` → http://127.0.0.1:8765 ; CLI: `python -m evals.golden.ingest.labeling status|validate|render` |
| S0 | `venv/bin/python -m evals.ingest_metrics.run_s0 [--include-unverified] [--skip-corpus] [--mlflow]` → `data/results/s0/` ; MLflow store `data/mlflow/mlflow.db` |
| QA set | validate: `venv/bin/python -m evals.golden.qa.validate --final` |
| Retrieval stage | `venv/bin/python -m evals.run --config evals/configs/s1_chunkers.yaml --split dev [--runs C2-dense] [--limit 3] [--no-mlflow]` (cold: ~15 min for C1–C5, C5 slowest; cached: seconds). `--limit` = smoke run, tagged SMOKE, report refuses it |
| Report | `venv/bin/python -m evals.report --stage S1 --split dev` |
| MLflow UI | `venv/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db` |

Tools on the Mac: Chrome 156 (HTML→PDF rendering), Tesseract 5.5.1, Apple M3 GPU (MPS), 16 GB RAM.

## 6. Code map

```
evals/corpus/      sec_client, edgar_discover, manifest(.csv), render (Chrome), fetch, cfpb, scans, *_candidates/selection/scan_specs.json
evals/parsers/     config (P1/P2/P1V/P2V, cache keys, downstream_config), docling_runner (venv-docling), docling_doc (cache → per-page views)
evals/golden/ingest/  schema (labels + F4 validator), labeling (select/prefill/render), app/ (server + index.html), pages/, images/, selection.json
evals/ingest_metrics/ normalize, text, geometry, tables (incl. TEDS), layout, confidence, run_s0
evals/chunking/    base (Chunk/Span provenance), source (parse → cleaned SourceDoc, drop_pages), c1–c5, metrics (§8.2), run_metrics, r2_report
evals/golden/qa/   schema + validate (F3), pages/pairs (job selection), draft (gpt-oss-120b), review_app/, finalise, questions.jsonl
evals/llm/         ledger (D7 spend + 190k tokens/day guard)
evals/retrieval/   metrics (§8.3), index_cache, views (scan swap)
evals/retrievers/  __init__ (registry RETRIEVERS), dense (R-dense), bm25 (R-bm25 + tokenizer), hybrid (R-hybrid, rrf), parent_child (R-pc, ParentMap)
evals/run.py  evals/report.py  evals/tracking.py  evals/configs/s1_chunkers.yaml
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
- **Import faiss only after torch** (macOS): if faiss loads first, the next MiniLM encode segfaults (two OpenMP runtimes). `evals/retrievers/dense.py::_faiss()` imports torch first — reuse it for any new FAISS code (HNSW/IVF in S3).
- Production code is imported as `app.…` (path set by `evals/parsers/production_adapter.py`), not `fastapi_service.app.…`.
- `value_in` (entity match) treats `62500.00` ≠ `$62,500` — numeric tokens must match exactly after normalization.
- Groq JSON mode fails (`json_validate_failed`) on raw control chars in CFPB text; `draft.py` salvages `failed_generation`.
