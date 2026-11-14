# Session Log

Dated record of working sessions: what changed, what was decided, and why. Newest first.
Add a new `## YYYY-MM-DD` section per session. Decisions get an ID (`S-<date>-<n>`) so later docs can cite them.

Related: initiative scope `loandoc-retrieval-evals-scope-v2.md` (its own Decision Log holds the D10 implementation choices) · R0 details `docs/r0_findings.md`.

---

## 2026-11-13 — R2 step 4: C5 semantic chunker

**Branch:** `feature/r2-chunkers`. Kousik: "go c5" (C2 overlap and C4 run-in heading questions still open; both left as built).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-1 | C5 = C2's sentence units, each embedded with MiniLM; cut where adjacent cosine distance > the document's 95th percentile; no overlap | Scope §6.1, literal | scope Decision Log |
| S-2026-11-13-2 | C5 max 200 tokens (C2), over-max segments split at their largest distance; min 50 (C4's, since C2 has none), small segments merged into the more similar neighbour if it fits | "Same min/max as C2"; C2 has no min | same |
| S-2026-11-13-3 | **Finding:** only 10% of C5 boundaries (1,108) pass the p95 threshold; 89% (9,544) come from the 200-token max split, which still cuts at the segment's largest distance. Median per-document threshold 0.96 (p5 0.83, p95 0.99): MiniLM sees most adjacent legal sentences as far apart | Long documents, p95 = 1 cut per 20 sentences, segments far over 200 tokens | open question below |

Changes: `evals/chunking/c5_semantic.py`; `check_corpus.py` (C5, exact checks); `evals/tests/test_chunking_c5.py` (10 tests, one with real MiniLM). Evals 221 pass; production untouched.

Corpus run (92 docs, P2/P2V, cleaner v2; **0 errors, no empty span, every non-blank page character covered**; 163 s on CPU incl. embedding):

| Chunker | Chunks | MiniLM tokens p5 / p50 / p95 / max | Over 254 | Non-blank page chars in no chunk |
|---|---|---|---|---|
| C5 | 10,744 | 54 / 139 / 200 / 200 | 0% | 0% |

C5 detail: 319 chunks under 50 tokens (no neighbour merge fit); 2,127 contain a split over-long sentence.

### Open items
- Kousik: keep C5 literal (p95, per-sentence embeddings — mostly size-driven cuts), or tune it (e.g. lower percentile, or embed each sentence with ±1 neighbour as in common semantic chunkers) so the threshold drives more cuts? Tuning = deviation from §6.1.
- Still open: C2 overlap (step 2), C4 run-in headings (step 3).
- Next: R2 step 5 (§8.2 chunk metrics on the corpus) — ask Kousik first.

---

## 2026-11-12 (cont.) — R2 step 3: C4 layout-aware chunker

**Branch:** `feature/r2-chunkers`. Kousik: "c4 please" (C2 overlap question from step 2 still open; C2 left as built).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-24 | `SourceDoc` gains Docling layout elements located in the cleaned page text (19,459/19,459 found) | C4 needs layout classes with provenance | scope Decision Log |
| S-2026-11-12-25 | C4 heading levels inferred from numbering (Docling: all level 1, no titles); unnumbered heading after a heading is joined to it; headings on ≥ 3 pages demoted to body; body-less headings kept as heading-only pieces | See Decision Log 2026-11-12 (C4 rows) | same |
| S-2026-11-12-26 | C4 sizes: ≤ 254 content tokens incl. prefix, prefix ≤ 64, min 50 (merge), C2 sentence windows inside sections; tables split by rows with header repeated; page furniture left out | Scope §6.1 | same |
| S-2026-11-12-27 | Table row spans trimmed of U+FEFF (provenance only; N2 re-checked: 4,774 chunks, 0 differences) | Cell text ending in a BOM gave padded spans | `source.py` |
| S-2026-11-12-28 | **Finding:** most numbered sections in the credit agreements are **not** Docling headings: of 1,311 EDGAR elements starting like "2.11 Title." / "Section 2.11 Title.", 738 are body `text`/`list` items and 573 are `section_header`. C4, built strictly on Docling headers as the scope says, misses those boundaries | Docling labels a run-in heading ("Section 2.11 Prepayments. (a) The Borrower may…") as a paragraph | open question below |

Changes: `evals/chunking/c4_layout.py`; `base.py` (`Element`, `strip_blank`); `source.py` (`locate_elements`, row-span trim); `stream.py` (`stream_from_pieces`, page offsets); `c2_sentence_window.py` (`sentence_windows` reused by C4); `check_corpus.py` (C4, word check, coverage); `evals/tests/test_chunking_c4.py` (24 tests). Evals 211 pass; app 173 pass / 2 skipped (production untouched).

Corpus run (92 docs, P2/P2V, cleaner v2; **0 errors, no empty span, no chunk without a span**):

| Chunker | Chunks | MiniLM tokens p5 / p50 / p95 / max | Over 254 | Non-blank page chars in no chunk |
|---|---|---|---|---|
| C1 | 4,774 | 380 / 476 / 711 / 1,954 | 99.0% | 0% |
| C2 | 9,760 | 77 / 176 / 200 / 200 | 0% | 0% |
| C3 | 6,583 | 253 / 254 / 254 / 254 | 0% | 0% |
| C4 | 8,503 (8,132 prose, 371 table) | 74 / 213 / 254 / 254 | 0% | 0.28% (headers/footers/page numbers only) |

C4 detail: 223/223 tables in chunks, 75 split by rows (header repeated in all pieces); 106 chunks under 50 tokens (83 prose that had no room to merge, 23 small tables).

### Open items
- Kousik: add a C4 rule that promotes run-in "Section 2.11 Title." paragraphs to headings (a deviation from "boundaries from Docling elements", so asking), or keep C4 strict and report the gap?
- Still open from step 2: C2 overlap (whole-sentence only vs token fallback).
- Next: R2 step 4 (C5 semantic) — ask Kousik first.

---

## 2026-11-12 (cont.) — R2 step 2: C2 sentence window, C3 fixed tokens

**Branch:** `feature/r2-chunkers`

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-20 | Sentence splitter **syntok 1.4.4**; token counts: MiniLM's own tokenizer + **tiktoken 0.14.0 `cl100k_base`**. Both installed in `venv/` without changing other packages, pinned in `requirements-dev.txt` | pysbd broke "Section 2.11(a)."; blingfire x86-only; nltk needs data | scope Decision Log |
| S-2026-11-12-21 | C3 = 254 content tokens (+[CLS]/[SEP] = 256), 32 overlap, word-snapped. C2 = whole sentences ≤ 200 tokens, ≤ 40 tokens of whole-sentence overlap, over-long sentences split C3-style. Both chunk the page stream across pages | Scope §6.1; no chunk truncated by MiniLM | same |
| S-2026-11-12-22 | Invisible format chars (U+FEFF) treated as blank in all spans; C1 spans trimmed (text unchanged) | Found by the corpus run: BOM-only/BOM-edged spans in `deal_05_ca` | same |
| S-2026-11-12-23 | **Finding:** C2 overlap often empty: 46% of boundaries have none (last sentence > 40 tokens); 23% of C2 chunks contain a split sentence | Long legal sentences; may matter for §8.2 sentence-break and evidence fragmentation | open question below |

Changes: `evals/chunking/tokens.py`, `stream.py`, `c2_sentence_window.py`, `c3_fixed_tokens.py`, `check_corpus.py` (corpus correctness check), `base.py` (`is_blank`, `has_text`, `SourceDoc.trim`), C1 span trimming; `evals/tests/test_chunking_c2_c3.py` (22 tests). Evals 187 pass; app 173 pass / 2 skipped (production untouched).

Corpus run (`venv/bin/python -m evals.chunking.check_corpus`; P2 digital / P2V scans, cleaner v2; 92 docs, 0 missing, **0 errors, no empty span, every non-blank page character covered by C2 and C3**):

| Chunker | Chunks | MiniLM tokens p5 / p50 / p95 / max | Over 254 |
|---|---|---|---|
| C1 | 4,774 | 380 / 476 / 711 / 1,954 | 99.0% |
| C2 | 9,760 | 77 / 176 / 200 / 200 | 0% |
| C3 | 6,583 | 253 / 254 / 254 / 254 | 0% |

N2 re-checked after the C1 span change: 4,774 chunks, 0 differences.

### Open items
- Kousik: keep C2's whole-sentence overlap (often none), or allow a token-level overlap when the last sentence is > 40 tokens?
- Next: R2 step 3 (C4 layout-aware) — **ask Kousik first**.

---

## 2026-11-12 (cont.) — R2 step 1: chunker interface, C1 with provenance, N2

**Branch:** `feature/r2-chunkers` (from `feature/r1-docling-adapter`; R1 follow-up is PR #5, open)

| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-18 | R2 done one module at a time: (1) interface + C1 + N2, (2) C2 + C3, (3) C4, (4) C5, (5) §8.2 chunk metrics on the corpus | Kousik's step-by-step preference | this log |
| S-2026-11-12-19 | Provenance = spans into cleaned page text; C1 = evals mirror of production with tagged lines; production chunker unchanged | See scope Decision Log 2026-11-12 | `evals/chunking/` |

Changes: `evals/chunking/base.py` (Span, SourceDoc, Chunk, Chunker), `source.py` (cache → adapter → cleaner v2 → SourceDoc with table-row spans), `c1_line_window.py`, `evals/tests/test_chunking_c1.py` (10 tests). N2 on the corpus: 92 docs, 4,774 chunks, 0 differences. 165 evals tests pass.

### Open items
- Next: R2 step 2 (C2 sentence window, C3 fixed tokens) — needs the MiniLM tokenizer.

---

## 2026-11-12 (cont.) — Docling → production adapter; cleaning and validation metrics

**Branch:** `feature/r1-docling-adapter` (from `main` after PR #3 merged)

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-6 | R1 merged to `main` via PR #3 (Kousik: "push to main"); leftovers continue on a new branch | Kousik's request; branch → PR → main rule kept | PR #3 |
| S-2026-11-12-7 | Adapter mapping and confidence rule (see scope Decision Log 2026-11-12) | Lets cleaner/validator (and R2's chunker) run unchanged on Docling output | `evals/parsers/production_adapter.py` |
| S-2026-11-12-8 | **Scope said the cleaner has 7 fixes; the code has different ones.** Scope corrected; metrics count the real fixes | Found while reading `cleaner.py` | scope §4 line; `evals/ingest_metrics/cleaning.py` |
| S-2026-11-12-9 | **Finding:** number-space repair corrupts multi-value Docling table cells (23 cells P2, 4 P2V) | Textract-era assumption | `docs/20_s0_results.md` §4.2 |
| S-2026-11-12-10 | **Finding:** validator fails every table-less document (`NO_STRUCTURED_DATA` ERROR): 27/41 EDGAR, 21/40 scans; required fields are for the Indian loan form | Textract-era assumption; recorded, not enforced | same |
| S-2026-11-12-11 | **Finding:** 3 of 40 scans produce no text with Docling + Tesseract (`scan_27/32/38`); Vision recovers 1. Not investigated | Shows up as NO_PAGES | same |

Changes: `evals/parsers/production_adapter.py`, `evals/ingest_metrics/cleaning.py`, `evals/ingest_metrics/run_cleaning.py`, cleaning tables in `s0_report.py`, `evals/tests/test_production_adapter.py` (7 tests). 148 evals tests pass. Run: 1.5 s for 92 docs × P2 + 40 × P2V.

Committed and pushed as `c0d3022`.

| ID | Decision | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-12 | **Cleaner v2** in production `cleaner.py`, opt-in via `version="v2"`: skips number-space repair on table cells. Default **v1** unchanged (production behaviour identical; `CleanedResult.cleaning_version` added). Evals run v1 and v2; **downstream evals use v2** | Kousik: "fix the cleaner". Scope rule: versioned config, not a silent edit. v2: 0 corrupted cells (v1: 27), nothing else changes | `fastapi_service/app/pipeline/cleaner.py`; scope Decision Log; `docs/20_s0_results.md` §4.2 |

Tests: app 173 passed / 2 skipped (4 new version tests); evals 150 (2 new: counters vs real cleaner, v2).

### Empty scans investigated (S-2026-11-12-13, -14)
Two independent causes; 8 of 40 simulated scans are affected.

| ID | Finding | Evidence |
|---|---|---|
| S-2026-11-12-13 | **3 EDGAR agreements render as blank pages.** `deal_03_ca` (Raymond James), `deal_07_ca` (Harmonic), `deal_11_ca` (TrueBlue) are filed on EDGAR as page images (`<IMG src="...001.jpg" title="slide1">`, 104–141 per doc) with an invisible **white** OCR text layer on top. Our renderer strips every `<img>` (D10), so all 353 pages are visually blank and keep only the white text (23% of all EDGAR characters). Consequences: scans `scan_27`, `scan_31`, `scan_32` are scans of blank pages; golden pages `deal_07_ca_p055` and `deal_11_ca_p022` have blank images (0 dark pixels) and were verified as near-empty; EDGAR digital metrics on these docs score Docling against invisible text | `evals/corpus/render.py` `sanitize_html`; source HTML img/white-style counts; rendered pixel check |
| S-2026-11-12-14 | **Docling trusts Tesseract orientation detection without a confidence threshold.** On heavy scans Tesseract OSD wrongly reports 180°/270° (confidence 0.1–3.0); Docling rotates the page and OCRs it upside down → garbage or nothing. Hits `scan_03`, `scan_23`, `scan_26`, `scan_35` (word recall 0.01–0.04) and `scan_38` (0 words); `scan_15` partly. P1/P2 only — Apple Vision (P1V/P2V) doesn't use OSD. Docling 2.147 has no option to disable OSD | Docling `tesseract_ocr_cli_model.py` L354–378; manual `tesseract --psm 0 -l osd` per scan; S0 per-page recall |

Not affected: the S0 P1-vs-P2 decision and H10 (no golden table page comes from these docs or scans). Affected: P1/P2 scan text metrics, H6 pairs, EDGAR text metrics on the 3 docs, 2 golden digital_prose pages, and any QA questions drawn from the 3 docs later.

### Fixes (Kousik chose: re-render with images, 0° shim, replace golden pages) — see scope Decision Log 2026-11-12
- Done: image filings rendered with their 350 page images (`evals/corpus/render.py`, `fetch.py --only`); scans rebuilt (only scan_27/31/32 changed); `tesseract-no-osd` shim + `ParserConfig.ocr_orientation`; golden pages replaced (`labeling replace`) → `deal_02_ca_p058`, `deal_05_am2_p001`; 5 tests (155 evals pass).
- Committed/merged as PR #4 with the re-parse still running.

### Re-run after fixes (later the same day)
| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-15 | 2 replacement golden pages pre-filled from P1 and **checked against the image by the agent** (`labeled_by: agent`); `deal_05_am2_p001`: title and recital "A." classes fixed | Kousik asked the agent to continue | label notes |
| S-2026-11-12-16 | **S0 re-run** on the fixed parse: decision (P2) and H10 unchanged; H6 still not supported; scan text much better (P2 CER 5.2 → 3.1%, numeric fidelity 72.6 → 84.1%) | Fixes don't touch golden tables | `docs/20_s0_results.md` §4.3 |
| S-2026-11-12-17 | **Finding, not fixed:** `scan_03` — Docling layout labels the whole page a *picture*, OCR text is stored inside it and dropped by the reader (as in Docling's own export). Reader left as is | Only page materially affected; changing the reader would pull logo text into CFPB pages and shift the golden pre-fill | same |

Re-parse: P1 and P2, 92 docs each, 0 failures. 155 evals tests pass.

### Open items
- R1 is complete apart from the split-table gap. Next: follow-up PR, then ask Kousik before R2.
- Then: R1 leftovers done → PR; ask before R2.

---

## 2026-11-12 — Golden-set tables settled by the agent; S0 run

**Branch:** `feature/r1-corpus-parsing`

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-12-1 | **Agent (Claude) settles the 13 disputed P1/P2 tables and types the from-scratch tables from the page images**, instead of Kousik | Kousik's request (no time to check each table). Disclosed in the results; every touched table is marked `settled_by: "agent"` | `evals/golden/ingest/settle_tables_agent.py`; label files; `docs/20_s0_results.md` §4.0 |
| S-2026-11-12-2 | Settling rules: grid follows the image; start from P1/P2 when one has the right structure (`adjudication` = P1/P2), else type it (`own`); fix text that differs from the image, including OCR misreads and text-layer content not visible on the page | Labels must match the image (README) | same |
| S-2026-11-12-3 | `cfpb_le_h24b_p003`: the single table entry was on the page heading (e1). e1 reclassified as `section_header`; 10 tables (e2–e11) typed | Labelling error found while settling | label file notes |
| S-2026-11-12-4 | **S0 decision: P2** (cell accuracy P2 − P1 = +13.0 pts, CI 2.0–24.2, n = 32 tables). **H10 supported** (+16.4, CI 3.6–33.6, n = 9). **H6 not supported** (with disclosure) | Pre-registered rules in `docs/20_s0_results.md` §2 | `docs/20_s0_results.md` §4 |
| S-2026-11-12-5 | Results tables generated from `summary.json` by `evals/ingest_metrics/s0_report.py` | F7 (no hand-typed numbers) | `evals/ingest_metrics/s0_report.py` |

Settled: P2 grid 7, P1 grid 3, typed 3. 141 evals tests pass (3 new: settle grid helper, report renderer).

### Open items
- Kousik (optional): spot-check the agent's grids in the app (Compare tab; `settled_by` in the label files).
- R1 remaining: cleaning/validation metrics need the Docling → `CleanedResult` adapter (shared with R2); open the R1 PR; ask before R2.

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

### S0 first run: P1-vs-P2 comparison invalid (pre-fill bias)
| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-06-7 | **Finding:** first S0 run on verified labels said P1 beats P2 on cell accuracy by 39 pts (CI −57…−23, 20 tables); H10 "not supported". **Not reported as a result:** 0 of 608 pre-filled cell texts were changed in verification, so labels = P1 output and P1 scores 1.0 on every table by construction. The 3 from-scratch tables have empty cells, so they score nothing | Pre-fill bias (scope §13) happened | `docs/20_s0_results.md` (after re-run) |
| S-2026-11-06-8 | **Settle the 9 pages / 13 tables where P1 and P2 disagree** (Kousik's choice of 3 options; others: re-label all ~600 cells, or call S0 inconclusive → P1 by tie rule). Kousik picks the grid that matches the image (P1 / P2 / own) and fixes text; fills the 3 empty from-scratch tables without Docling output. Choice stored per table as `adjudication`. Cells both parsers read identically stay as pre-filled: residual bias, stated in the report | Cheapest fix that removes the by-construction P1 advantage | Scope Decision Log |
| S-2026-11-06-9 | Labelling app: **Compare tab** (P1 and P2 grids side by side, differing cells in yellow, "Use P1/P2 grid", "Keep my grid"); ⚖ chip on pages to settle; `GET /api/compare/<page>`; `evals/golden/ingest/app/compare.py` + 6 tests. Hidden on from-scratch pages | Tool for S-2026-11-06-8. Tested in headless Chrome (render, use-grid, undo; nothing saved) | `evals/golden/ingest/app/` |

138 evals tests pass. S0 decision and H10 wait for the re-run after settling.

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
