# LoanDoc Retrieval & Evals Initiative — Scope v2

**Status:** Draft · **Owner:** Kousik Nandury · **Date:** 2026-11-04
**Supersedes:** `loandoc-retrieval-evals-scope.md` (v1, 2026-11-01/03). v1 is kept for history; build from this file.
**Repo:** github.com/nkousik18/FinanceHQ (internal name FinanceHQ, public name LoanDoc)
**Naming:** This is the "Retrieval & Evals" initiative. Milestones are numbered **R0–R8** so they don't collide with any existing phase or milestone names in the repo's roadmap. If the repo has a roadmap file, add a section for this initiative there rather than renumbering anything.

## 0. What Changed From v1

The goal, the metrics catalog (§8), the stages, the statistics and the hypotheses carry over. What changed is the environment and the data:

| Area | v1 | v2 | Why |
|---|---|---|---|
| Cloud | AWS: Textract, S3, EC2, MLflow artifacts in S3 | **No AWS.** Everything runs locally | AWS free tier expired; the account's backend was deleted |
| Parser | AWS Textract (TABLES + FORMS, + LAYOUT re-extraction) | **Docling** (local, open source), behind a parser interface | No AWS; $0 per page; also makes a later parser comparison a config change |
| Corpus | ~60 private loan PDFs (not actually in the repo; only `loan1.pdf` exists) | **Public documents:** SEC EDGAR credit agreements with their amendments, CFPB consumer loan forms, simulated scans | Reproducible, showable, and harder (long legal documents, amendments) |
| Private-data rules (v1 D11, F9, F10) | Corpus, golden sets and examples must never enter the repo; CI on synthetic docs | **Relaxed.** Golden sets, labels and examples are public and committed. CI runs on a small real public subset | The corpus is public |
| Storage | S3 (Block Public Access) | Local gitignored data directory + committed manifest and fetch script | No AWS |
| MLflow | Self-hosted server, artifacts in S3 | Local MLflow (SQLite backend + local artifact folder) | No AWS |
| Cost caps | $15 Textract re-extraction; $40 LLM | **Parsing: $0, time-capped** (D6). LLM caps unchanged ($40) | Docling is free but slow on CPU |
| §4 facts | Several were wrong (see §4) | Corrected against the code on 2026-11-04 | — |

## 1. Problem
LoanDoc is a working RAG system: document extraction, one line-window chunker, dense FAISS search, Groq generation. Its evaluation is weak:
- 7 questions over one synthetic document
- "Groundedness" is a keyword match, and "relevance" is word overlap between question and answer
- **Retrieval is never measured.** A wrong answer can't be traced to a bad chunk or a bad model
- **Parsing is never measured.** Nobody knows how accurately the parser reads the tables, forms and layout of these documents, or whether its confidence scores mean anything
- **Latency is measured only end to end.** Time-to-first-token and token throughput have never been recorded, even though the streaming endpoint already streams real tokens
- **The corpus is one PDF.** No claim made on it generalizes

AI engineering postings ask for grounded, cited answers (36 of 166 saved postings), vector search (33), and eval sets and pipelines (28), along with measured cost and latency. Few name a specific retrieval technique (hybrid 5, rerank 3, graph 4, hierarchical 1). A plain RAG app with no measurement doesn't meet that bar.

## 2. Goal
Build an offline evaluation pipeline that measures every stage of LoanDoc:
- **ingestion and parsing** against a labeled page set
- **chunking**, with metrics computed from the chunks themselves
- **retrieval** against a span-labeled question set
- **generation**: quality, latency and cost

Each comparison comes with statistical confidence intervals. Use the pipeline to run a staged comparison of 5 chunkers and 5 retrieval strategies, plus a load test for latency. Then report which configuration wins on this data, by how much, at what cost and latency, and which stage causes each remaining wrong answer.

All of it runs on a **public corpus of real loan documents**, so every golden-set item, label, example and result can be published and reproduced by anyone with the repo.

The experiment and the eval pipeline are one system. The comparison is the first thing the pipeline is used for.

## 3. Non-goals
- **No embedding-model comparison.** all-MiniLM-L6-v2 stays fixed as the control variable (see R0 for its truncation check). Swapping embedders is a possible follow-up.
- **No parser bake-off.** Docling is the one parser measured. S0 compares two Docling settings only (§9.3). Comparing other parsers (Textract, Marker, MinerU) is a possible follow-up; the parser interface (§6.0) makes it a config change.
- **No changes to the production app's extraction or storage.** Production still imports Textract and S3 code and cannot currently run without AWS. Porting production to Docling and local storage is a follow-up; R8 recommends how.
- **No graph RAG.** Loan documents are mostly independent files, so the cost is high for little expected gain.
- **No deployment.** No cloud hosts exist now. R8 produces a written rollout recommendation only, including whether the winner fits a small (1GB) host.
- **No online A/B testing.** There is no live traffic. The existing epsilon-greedy prompt selector stays as it is and is not part of this work.
- **No LLM fine-tuning, no self-hosted model serving.** Those belong to the separate inference module.
- **No new vector database.** FAISS stays in-process. Every retriever must run inside the same process with no new services.
- **No UI changes,** and no change to the production streaming endpoint.

## 4. Current State (verified against the code on 2026-11-04)
- **Extraction (production):** AWS Textract `AnalyzeDocument` / `StartDocumentAnalysis` with TABLES + FORMS (`app/pipeline/extractor.py:35`). Output objects include `ExtractionResult`, `PageResult` and `Table` (`Table.to_markdown()`). **The AWS account backing it is gone;** evals use Docling instead (§6.0).
- **Cleaning:** 7 ordered fixes: number-space repair, repeated header/footer removal, empty-checkbox artifact removal, duplicate-line removal, whitespace normalization, hyphenation repair, smart-quote/em-dash normalization. Written for Textract output; R1 measures how it behaves on Docling output (over-cleaning rate, §8.1).
- **Validation:** per-block confidence (warn < 85, error < 70), required fields, page count. `WARNING` logs and continues; `ERROR` fails the document.
- **Chunking:** **sliding window over lines**, respecting markdown section headings (`app/pipeline/chunker.py`). Size and overlap are set in "tokens" but measured as **word count ÷ 0.75** — no tokenizer, no sentence splitter. Defaults: size 400, overlap 60. *(v1 said nltk sentences + tiktoken; that was wrong.)* `chunk_id = "{session_id}-{index}"`.
- **Embedding:** sentence-transformers `all-MiniLM-L6-v2`, 384-dim, CPU.
- **Index:** FAISS `IndexFlatIP` on normalized vectors, default `top_k=5`. **No 0.25 score threshold exists in the code.** *(v1 assumed one.)*
- **Context assembly:** `MAX_CONTEXT_WORDS = 1200` (`app/prompts/router.py:64`), trims lowest-ranked chunks first.
- **Generation:** Groq `qwen/qwen3.8-27b` (since 2026-11-04; replaces the retired `llama-3.3-70b-versatile`, see `docs/r0_findings.md` §3), temperature 0.1, max_tokens 1024. Five intent templates × two variants. Set in `app/core/config.py` and `.env`.
- **Streaming:** `/query/stream` uses **native Groq streaming** (`stream=True`) relayed as SSE. *(v1 said it was simulated; that was wrong.)* The eval harness still times tokens client-side against the Groq API directly (§9.5) so retrieval and network effects are separated.
- **Tracking:** MLflow. Previously a self-hosted server with S3 artifacts; now local.
- **Corpus:** one PDF in the repo (`loan1.pdf`).
- **Tests:** after R0, 143 passing and 2 stale modules skipped with reasons (`docs/r0_findings.md` §5).
- **Known bug (fixed in R0):** `evict_session()` was never called after a new document was indexed, so queries hit the stale cached index until restart.
- **Truncation (measured in R0):** on `loan1.pdf`, 100% of C1 chunks exceed MiniLM's 256-token limit and 22.5% of the document's words are never embedded (`docs/r0_findings.md` §2).

## 5. Corpus and Golden Sets

### 5.0 Corpus (public)
Three parts, all public:

| Part | Content | Target size | What it exercises |
|---|---|---|---|
| **EDGAR deals** | 12 corporate credit agreements filed with the SEC (Exhibit 10), each with 1–3 of its amendments. One **deal** = an agreement plus its amendments | ≤ ~2,500 pages total | Long legal prose, defined terms, pricing grids, covenant tables, amendments that override the original |
| **Consumer forms** | CFPB sample Loan Estimate and Closing Disclosure forms, plus other public-domain federal loan forms if needed to fill table and form quotas | 8–12 documents | Dense fee tables, form fields, complex multi-region layout |
| **Simulated scans** | 40 pages drawn from the two parts above, rendered to images with controlled degradation (DPI, blur, noise, skew, JPEG), saved as image-only PDFs | 40 pages | OCR path, confidence, scanned-page CER (H6) |

**Selection rules for EDGAR deals:**
- Filed 2015 or later; a mix of borrower sizes and industries; at least 3 deals with 2+ amendments.
- Prefer filings that include a PDF. EDGAR exhibits are mostly HTML; when no PDF exists, render the HTML to PDF with headless Chromium at fixed settings, and mark the document `source_format: html_rendered`. Report every metric split by `source_format` where it could matter (S0 especially).
- Skip agreements above 300 pages unless needed for a quota.
- Kousik approves the final deal list before any download (~30 min).

**Manifest (committed):** one row per document: `doc_id`, `part`, `deal_id` (EDGAR only), source URL, CIK and accession number (EDGAR), filing date, `source_format`, page count, sha256. A `fetch` script downloads (and renders if needed) every document from the manifest and verifies the checksums. Raw PDFs are not committed (size); see D11.

**EDGAR access:** follow the SEC's fair-access rules — at most 10 requests per second, and a descriptive User-Agent. The User-Agent string comes from an environment variable (`SEC_USER_AGENT`) that Kousik sets; it is never hard-coded.

**Sessions:** each EDGAR deal is one session (agreement + amendments). Consumer forms are grouped into sessions of related forms (e.g. a Loan Estimate with its matching Closing Disclosure) where such pairs exist, else one document per session.

**Simulated scans are labeled as such everywhere.** They are a controlled stand-in for real scans: the original text layer gives an exact reference, which real scans lack. Real scanned loan documents are a nice-to-have (Open Question 1).

There are two golden sets:
- **QA golden set** (§5.1–5.3): questions with span-labeled evidence. Used for retrieval and generation metrics.
- **Ingestion golden set** (§5.4): hand-labeled pages. Used for parsing metrics.

### 5.1 QA set: size and mix
120 questions in total, split before any experiment runs:
- **dev (40):** used to pick winners at each stage
- **test (80):** run once at the end (stage S4) and used for every reported number

| Type | Count | What it tests |
|---|---|---|
| `lookup` | 25 | A single fact in running text (rate, maturity date, lender name, a defined term) |
| `table` | 25 | A value that only exists inside a table (pricing-grid margin, fee in a Closing Disclosure table) |
| `calculation` | 20 | Arithmetic over document values (total commitments, cash to close, margin after a step-down) |
| `cross_doc` | 15 | Needs evidence from 2+ documents in the same session. **At least 10 are amendment questions**: the answer is the value after an amendment changed it |
| `explain` | 15 | A clause or condition in prose (prepayment terms, an event of default, a covenant) |
| `unanswerable` | 20 | Plausible questions the documents don't answer |

Questions are spread across the corpus: at least 70% on EDGAR deals, at least 20% on consumer forms, and at least 8 whose evidence is on a simulated-scan page (those pages are indexed in their scanned form for the run). The dev/test split keeps the type mix proportional.

### 5.2 QA set: schema (JSONL, one question per line)
```json
{
  "qid": "q0042",
  "split": "test",
  "type": "cross_doc",
  "question": "What is the applicable margin for SOFR loans at the lowest pricing level after the Second Amendment?",
  "session_id": "deal_07",
  "session_docs": ["deal_07_ca", "deal_07_am1", "deal_07_am2"],
  "evidence": [
    {"doc_id": "deal_07_am2", "page": 3, "span_text": "..."}
  ],
  "reference_answer": "...",
  "key_points": ["..."],
  "entities": ["..."],
  "superseded_values": ["..."],
  "answer_numeric": null,
  "numeric_tolerance": null,
  "unanswerable": false,
  "provenance": "llm_drafted_human_verified",
  "verified_by": "kousik",
  "verified_on": "2026-11-xx"
}
```
- `qid` is permanent. Questions are never renumbered or edited in place. A fix creates a new `qid` and marks the old one `retired: true`.
- `evidence` is labeled by **page and span text, never by chunk ID**. Chunk IDs change with every chunker, and span labels let one golden set score all of them.
- `span_text` must appear in the **cleaned** page text of the parser configuration chosen in S0. A validator checks this (§10, F3). If S0 picks a different parser configuration than the one used for drafting, re-run the validator and fix any misses before S1.
- `key_points` are the facts a complete answer must contain (used for answer completeness).
- `entities` are the amounts, rates, dates and party names in the reference answer (used for entity recall). The drafting script pre-fills them; Kousik verifies them with the question.
- `superseded_values` (new in v2, amendment questions only): values that an amendment replaced. Used for the stale-answer rate (§8.4).
- `unanswerable` questions have an empty `evidence` list.

### 5.3 QA set: how it's built
1. The agent writes a drafting script. It samples pages per document and asks an LLM for candidate questions by type, each with a proposed answer, evidence span, key points and entities. For amendment questions it pairs each amendment section with the original section it modifies.
2. The script auto-rejects any candidate whose span is not found verbatim in the cleaned text.
3. Kousik verifies every question that goes into the set: keep, edit or reject. **Nothing enters the set without his verification.** About 6 hours of his time.
4. The script assigns `qid`s and the dev/test split.

### 5.4 Ingestion golden set
**30 pages**, sampled from the corpus to fill these quotas:

| Stratum | Pages | Source |
|---|---|---|
| Digital, prose and forms | 10 | EDGAR prose pages; consumer form pages |
| Table-heavy (pricing grids, fee tables, commitment schedules) | 8 | EDGAR schedules; Closing Disclosure tables |
| Scanned or low quality | 6 | Simulated scans (§5.0) |
| Multi-column or complex layout | 6 | Consumer forms (multi-region layout); any multi-column EDGAR pages |

If the corpus doesn't have 6 genuinely multi-column or complex-layout pages, take what exists, fill the rest from the table-heavy stratum, and record the actual counts (Open Question 2).

**Labels per page (JSON, one file per page):**
- `page_id` (`doc_id` + page number), `stratum`, `source_format`
- `reference_text`: needed only for pages without a usable text layer. Simulated scans use the text layer of the page they were rendered from, so they need no hand-typed reference.
- `tables`: bounding box, cell grid (row, column, row span, column span, text), header row indices
- `key_values`: key text and value text
- `layout_elements`: bounding box and class (title, section header, text, list, table, page header, page footer, page number)
- `reading_order`: the layout element IDs in correct reading order
- `header_footer_lines`: lines that are page headers or footers

**How it's built:**
1. The agent writes a labeling script that **pre-fills** each page's labels from Docling's output and renders the page image beside them.
2. Kousik corrects every label. About 6 hours of his time.
3. To measure the bias pre-filling introduces (labelers tend to accept what's already there), Kousik labels **5 of the 30 pages from scratch** first, without seeing Docling's output. Report Docling's scores on those 5 separately from the other 25.

## 6. Parser and Chunkers

### 6.0 Parser interface (new in v2)
One interface, `parse(pdf_path, config) -> ExtractionResult`, returning the same `ExtractionResult` / `PageResult` / `Table` objects production uses, plus:
- layout elements with class, bounding box and reading-order index
- per-word confidence where an OCR engine produced the text, and any page-level confidence the installed Docling version reports (layout, OCR, table, parse scores)
- the parser name, version and config, recorded with every cached result

The Docling adapter is the only implementation built now. A Textract adapter can be added later for a parser comparison without touching anything downstream. Docling's layout labels are mapped onto the label classes in §5.4; the mapping is recorded in the decision log.

### 6.1 Chunkers
Every chunker implements one interface and emits chunks with **provenance**: `doc_id`, `page_start`, `page_end`, and character offsets into the cleaned page text. Retrieval metrics depend on provenance, so a chunker without it can't be evaluated.

| ID | Chunker | Rules |
|---|---|---|
| `C1` | Line window (baseline) | **Current production behavior, exactly:** line-based sliding window respecting markdown headings, size 400 / overlap 60 "tokens" measured as words ÷ 0.75 |
| `C2` | Sentence window, small | Sentence-split window, ~200 MiniLM tokens, 40 overlap. Tests smaller, sentence-aligned chunks, and fits inside MiniLM's input limit (R0) |
| `C3` | Fixed-size tokens | 256 MiniLM tokens, 32 overlap, no sentence awareness. Simplest comparison point |
| `C4` | **Layout-aware** | Uses Docling's layout elements and section hierarchy. Rules below |
| `C5` | Semantic | Splits where the embedding similarity between adjacent sentences drops below the 95th-percentile distance threshold for that document. Same min/max token bounds as C2 |

**C4 layout-aware rules:**
- Section boundaries come from Docling's title and section-header elements.
- Each chunk is prefixed with its heading path, e.g. `"Credit Agreement > Article II > 2.11 Prepayments: "`. The prefix is embedded, and it counts toward the token budget.
- Tables are kept whole as markdown (reuse `Table.to_markdown()`). A table over the max size is split by rows, and **the header row is repeated in every piece**.
- Form key-value pairs from the same section are grouped into one chunk.
- Max chunk size is 256 MiniLM tokens, and the minimum is 50. A prose section over the max falls back to sentence windowing inside that section.

Parent–child retrieval is a retrieval strategy (§7), not a chunker, because it needs two chunk levels.

## 7. Retrieval Strategies

| ID | Strategy | Details |
|---|---|---|
| `R-dense` | Dense (baseline) | Current FAISS `IndexFlatIP` cosine search |
| `R-bm25` | BM25 only | In-process BM25 over the same chunks. A lower bound, and useful for diagnosis |
| `R-hybrid` | Hybrid | Dense top-20 plus BM25 top-20, merged with reciprocal rank fusion (RRF, k=60) |
| `R-pc` | Parent–child | Search over small child chunks (C2 or C3 settings). Return each hit's parent: the C4 section if available, otherwise a ~800-token window. Remove duplicate parents |
| `+rerank` | Cross-encoder rerank | Applies on top of any strategy above. Take the top 20 candidates, rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep the top k |

**Index types (separate, small study):** compare `IndexFlatIP` against `IndexHNSWFlat` and `IndexIVFFlat` on the winning configuration. Report recall against Flat and query latency. At this corpus size Flat is expected to stay fast enough; the point is the curve, not a winner.

**Score thresholds:** production has none (§4), and none is added. Scores from dense, BM25, RRF and the reranker are on different scales anyway. Unanswerable questions are scored at the generation stage (§8.4).

## 8. Metrics

**How to read these tables:**
- **Source** says what a metric needs:
  - **Auto:** computed from data that already exists, no labels
  - **QA:** the QA golden set (§5.1)
  - **Ingest:** the ingestion golden set (§5.4)
  - **Judge:** the LLM judge, valid only after calibration (§8.6)
- **★ marks a headline metric.** Headline metrics are reported with confidence intervals. Every other metric is logged for diagnosis, without CIs and without being used to pick anything.
- **Only one metric picks the winner at each stage** (D3). With dozens of metrics, some will show a "win" by chance alone. A fixed decision metric and the pre-registered hypotheses (§11) guard against that.
- Every metric is also split by corpus part (EDGAR / consumer / simulated scan) where it applies.

### 8.1 Ingestion and parsing

**Parser confidence (Auto).** Docling gives no single per-block confidence the way Textract did. Use what exists:
- per-word confidence from the OCR engine on OCR'd pages (simulated scans, and any page Docling routes to OCR)
- Docling's page-level confidence scores (layout, OCR, table, parse), if the installed version reports them; confirm in R1 and record in the decision log

| Metric | Definition |
|---|---|
| Mean / p10 / min confidence | Per available score type (OCR word confidence; each Docling page-level score), reported per page and per document |
| Low-confidence word rate | % of OCR'd words below 90, and % below 80 |
| Low-confidence page count | Pages whose p10 OCR word confidence is below 80, or whose Docling page grade is the lowest grade |
| Document ingestion score | For OCR'd documents: % of words with confidence ≥ 90. For digital documents: the mean Docling page-level parse score. Used in §8.7; the two kinds are reported separately, never mixed in one tercile |

**Text accuracy.**
The reference text is the PDF's embedded text layer for digital pages (extracted with PyMuPDF or pdfplumber; Auto), the source page's text layer for simulated scans (Auto), and `reference_text` for any other page without a usable text layer (Ingest). Use a text layer only where it has at least 50 words and no more than 5% non-printable characters. Record each excluded page and the reason.

| Metric | Definition |
|---|---|
| ★ CER | Character edit distance between the parser's text and the reference, divided by reference length, after normalization (D10). Computed per page; reported as median and p90 across pages |
| WER | The same, counted in words |
| ★ Numeric fidelity | % of numeric tokens in the reference (amounts, rates, dates, percentages) that appear exactly in the cleaned output. Compared as multisets, so reading order doesn't affect it |
| Cleaning repair counts | How many times each of the 7 cleaning fixes fired, per document |
| Characters removed by cleaning | % of extracted characters the cleaner removed |
| Over-cleaning rate | Pages where CER after cleaning is worse than before cleaning. Matters more in v2: the cleaner was written for Textract output |

**Tables (Ingest).**

| Metric | Definition |
|---|---|
| Table detection precision / recall / F1 | A predicted table matches a labeled table when their boxes overlap with IoU ≥ 0.5 |
| ★ Cell accuracy | % of labeled cells whose normalized text appears in the matched predicted cell at the same row and column. **The decision metric for S0** |
| TEDS | Tree-edit-distance similarity between predicted and labeled table structure (0–1). Captures merged cells and row/column structure. Use an existing open-source implementation |
| Header-row accuracy | % of tables whose header rows were identified correctly |
| Split-table rate | % of labeled tables that span pages and come out as separate unlinked fragments |

**Forms / key-value pairs (Ingest).** Docling's key-value extraction is weaker than Textract FORMS. Measure it anyway; a low number is a finding, not a failure.

| Metric | Definition |
|---|---|
| Key-value precision / recall / F1 | Pairs matched on normalized key text |
| Value exact-match rate | Matched keys whose normalized value is exactly right |

**Layout (Ingest).**

| Metric | Definition |
|---|---|
| Element classification F1 | Per class and macro-averaged. A predicted element matches a labeled one at IoU ≥ 0.5 |
| ★ Section-header recall | % of labeled section headers detected as a title or section-header element. C4 depends on this |
| Reading-order accuracy | Kendall's τ between predicted and labeled element order, per page. Report multi-column pages separately |
| Header/footer removal precision and recall | Lines the parser or cleaner removed as headers/footers vs `header_footer_lines` |

**Throughput and reliability (Auto).**
- Pages per minute on the laptop CPU, by parser config and by part; stage latency p50/p95 for parsing, cleaning, validation, chunking and indexing
- Peak memory during parsing
- $ per page: $0 for Docling; record compute time instead
- Failure and retry rate by stage
- Validation warnings and errors per document

### 8.2 Chunking (Auto unless noted; computed per chunker with no queries)

| Metric | Definition |
|---|---|
| Chunk count; size distribution | p5 / p50 / p95 / max tokens per chunk, in both tiktoken and MiniLM tokens, plus words |
| ★ Embedder truncation rate | % of chunks longer than the embedder's max sequence length, and % of total text that never gets embedded |
| Table-break rate | % of tables split across chunks without the header row repeated |
| Sentence-break rate | % of chunk boundaries that fall mid-sentence |
| ★ Evidence fragmentation (QA) | % of QA evidence spans that no single chunk covers at ≥ 50% overlap |
| Overlap redundancy | % of indexed tokens that are duplicated because of overlap |
| Index size and build time | Memory, disk, and embedding time per document and in total |

### 8.3 Retrieval
A retrieved chunk **covers** an evidence span when it is from the same `doc_id`, its page range includes the span's page, and at least 50% of the span's normalized characters fall inside the chunk's offsets.

| Metric | Source | Definition |
|---|---|---|
| ★ `recall@budget` | QA | Fraction of evidence spans covered within the **1,200-word context budget**, after the same trimming production uses. **The decision metric for S1 and S2** |
| `recall@k` (k = 1, 3, 5, 10) | QA | Fraction of a question's evidence spans covered by the top k chunks |
| `hit@k` | QA | At least one evidence span covered in the top k |
| `precision@k` | QA | % of the top k chunks that cover any evidence span |
| Context precision | QA | Rank-weighted precision: the average of precision@i over each rank i that holds a relevant chunk (the RAGAS definition) |
| `mrr@10` | QA | 1 / rank of the first chunk that covers any evidence span |
| `ndcg@10` | QA | Binary relevance per chunk |
| `map@10` | QA | Mean average precision |
| ★ Entity recall | QA + Auto | % of the question's `entities` that appear in the retrieved context within the budget. Numbers, rates and dates are matched after normalization; names by normalized string match |
| Context noise ratio | QA | % of context words that come from chunks covering no evidence |
| Context redundancy | Auto | Mean pairwise overlap between retrieved chunks (shared tokens ÷ smaller chunk's tokens) |
| Cross-document coverage | QA | For `cross_doc` questions: % where every needed document contributed a covering chunk |
| Amendment coverage (new in v2) | QA | For amendment questions: % where the amending span is in the context budget. Also logged: % where only the superseded original span is |
| `context_words` | Auto | Average words passed to the generator |
| Retrieval latency p50 / p95 | Auto | Total, and per component: query embedding, vector search, BM25, fusion, rerank |
| Recall vs exact search | Auto | For HNSW and IVF: overlap of their top k with Flat's top k |

Unanswerable questions are excluded from all coverage-based metrics. Break every metric down **by question type** and **by corpus part** as well as overall.

### 8.4 Generation quality

| Metric | Source | Definition |
|---|---|---|
| ★ Faithfulness | Judge | % of the answer's claims supported by the retrieved context, graded claim by claim |
| Hallucination count | Judge | Unsupported claims per answer (a count, so one bad answer can't hide behind a high average) |
| ★ Answer correctness | QA + Judge | Correct / partial / wrong against `reference_answer`, with `partial` = 0.5. **The decision metric for S4** |
| ★ Numeric exact match | QA | For `calculation` questions: the parsed number is within `numeric_tolerance` of `answer_numeric`. Deterministic, no judge |
| Answer completeness | QA + Judge | % of `key_points` the answer contains |
| Answer relevance | Judge | Does the answer address the question asked (1–5 scale, rubric in the judge prompt) |
| Refusal correctness | QA | For `unanswerable` questions: the answer declines rather than inventing a value |
| False-refusal rate | QA | For answerable questions where the evidence was retrieved: the answer declines anyway |
| Stale-answer rate (new in v2) | QA | For amendment questions: % of answers that state a value from `superseded_values` as current. Deterministic, no judge |
| Citation precision | QA | % of `[chunk_id]` citations whose chunk covers an evidence span |
| Citation recall | QA | % of covering chunks in the context that the answer cites |
| Noise sensitivity | QA + Judge | Drop in answer correctness when 3 random non-evidence chunks are added to the context. Run on the dev split only, for the final top configuration |
| Format compliance | Auto | The answer follows its intent template's required structure (checked by a per-template validator) |
| Run-to-run consistency | Auto | Agreement rate across 3 runs of the same question at temperature 0. Run on the dev split only |

### 8.5 Generation latency and cost

| Metric | Source | Definition |
|---|---|---|
| ★ TTFT | Auto | Time from sending the request to receiving the first streamed content token |
| TTFS | Auto | Time from sending the request to receiving the first sentence-ending punctuation (`.`, `?`, `!` followed by whitespace or end of stream, ignoring decimal points in numbers) |
| Inter-token latency | Auto | Mean gap between streamed tokens after the first |
| Output tokens/sec | Auto | Output tokens ÷ (time of last token − time of first token) |
| ★ End-to-end latency | Auto | Question received to final token, including retrieval and context assembly |
| Input / output tokens | Auto | Per query, as reported by the provider |
| ★ $ per query, $ per 1k queries | Auto | Using the provider's published prices for the exact model, recorded with the run (even if the run itself used a free tier) |
| Input-token vs output-token cost | Auto | The split of each query's cost. Input tokens are billed in full when the request starts, so there is no separate "cost to first token"; this split shows whether context size or answer length is driving spend |
| ★ Cost per correct answer | QA + Judge | Total $ ÷ (sum of answer-correctness scores) |
| Error and timeout rate | Auto | Failed or timed-out generation calls |

**All timings are measured client-side with native provider streaming** (§9.5). Latency **percentiles** (p50 / p95) are reported from the load test (S5), not from the 80 test questions: p95 of 80 rests on about 4 data points, and p99 on less than one. S4 reports latency means and medians only.

### 8.6 Judge calibration (must happen before any judge-based number is reported)
1. Kousik hand-labels 50 answers drawn from the dev split, mixed across configurations, for faithfulness (each claim supported or not), answer correctness (correct/partial/wrong), and answer relevance (1–5).
2. Compute Cohen's κ between the judge and Kousik on faithfulness and correctness, and weighted κ on relevance.
3. **Requirement:** κ ≥ 0.6 on faithfulness and correctness. If either is lower, revise the judge rubric and re-test on a fresh 25 dev answers. Don't tune the rubric on test data. Relevance is reported with its κ but doesn't block.
4. Record the κ values, the rubric version and the judge model in the results.

### 8.7 End-to-end analyses (run after S4)
- **Quality / latency / cost chart:** answer correctness vs p95 end-to-end latency vs $ per 1k queries, for every S4 configuration. Mark which configurations no other configuration beats on all three.
- **Error attribution:** every wrong or partial S4 answer is assigned to the first stage that failed, using that stage's metrics for the question:
  1. **Parse:** an evidence span's numbers or text are wrong in the cleaned output
  2. **Chunk:** the evidence is fragmented across chunks
  3. **Retrieve:** the evidence wasn't within the context budget
  4. **Generate:** the evidence was in context and the answer was still wrong (including stale answers)

  Report counts and percentages per configuration, and per corpus part.
- **Ingestion confidence vs answer accuracy:** bucket test questions by their source document's ingestion score (terciles, computed separately for OCR'd and digital documents) and compare answer correctness across buckets. If low-confidence documents fail more often, that supports flagging such documents at upload time. Report it as exploratory, with n per bucket.

## 9. Pipeline

### 9.1 Layout
The agent picks paths that follow repo conventions. Logically:
- `evals/corpus/` — manifest, fetch script (download, HTML→PDF render, checksum), scan simulator. Committed
- `evals/parsers/` — parser interface and the Docling adapter
- `evals/golden/qa/` — schema, validator, drafting script, **and the QA golden set**. All committed
- `evals/golden/ingest/` — schema, validator, labeling script, **page labels and the 30 page images**. All committed
- `evals/fixtures/ci/` — the CI subset: 3–4 small public documents, their cached parser output, and a 20-question smoke set drawn from the dev split. Committed
- `evals/ingest_metrics/` — the §8.1 metrics
- `evals/chunkers/`, `evals/retrievers/` — implementations behind shared interfaces. Production code should import the same chunker and retriever classes, rather than keeping a copy for evals
- `evals/configs/` — YAML, one file per stage
- `evals/run.py` — CLI runner
- `evals/loadtest.py` — S5 load test
- `evals/report.py` — generates the results tables and charts
- `evals/baseline.json` — stored smoke-subset scores for the CI gate
- `data/` (gitignored) — downloaded PDFs, parser cache, chunk/embedding cache, local MLflow store

### 9.2 Runner
```
python -m evals.corpus.fetch --manifest evals/corpus/manifest.csv
python -m evals.run --config evals/configs/s0_ingestion.yaml
python -m evals.run --config evals/configs/s1_chunkers.yaml --split dev
python -m evals.run --config evals/configs/s4_generation.yaml --split test --confirm-cost
python -m evals.loadtest --config evals/configs/s5_load.yaml --confirm-cost
```
- **Config file:** lists chunker × retriever × rerank × top_k combinations explicitly, never as a Cartesian product generated automatically.
- **Parser cache:** parser output stored once per document and parser config, keyed by `hash(doc sha256, parser name, parser version, parser config)`. Experiments after S0 never re-parse.
- **Chunk/embedding cache:** keyed by `hash(chunker_id, chunker_params, parser_cache_key, cleaning_version, embedder)`. Re-running a stage re-embeds nothing that hasn't changed.
- **MLflow (local):** SQLite backend and a local artifact folder under `data/`. One run per configuration, in a new experiment, `loandoc_retrieval_evals`. S0 and S5 use their own experiments, `loandoc_ingestion_evals` and `loandoc_load_tests`.
  - Params: every config field, git SHA, corpus manifest hash, golden-set versions, parser name/version/config, model names, judge rubric version.
  - Metrics: everything in §8 that applies to the stage.
  - Artifacts: per-question (or per-page, for S0) results JSONL. For QA stages: qid, retrieved chunk IDs with ranks and scores, covered spans, entities found, answer, citations, judge grades with rationales, tokens, cost, and every timing in §8.5.
- **Cost guard:** stages that call an LLM print an estimated $ cost (questions × configurations × average tokens × price) and refuse to run without `--confirm-cost`. Caps in D7.
- **Parse-time guard:** S0 and any re-parse print an estimated CPU time (pages × measured seconds per page on a 10-page sample) before running. Cap in D6.
- **Determinism:** temperature 0 for generation and the judge in eval runs. Record model IDs exactly as the provider returns them. Pin the Docling version and its model weights.

### 9.3 Experiment stages
| Stage | Runs | Split | Picks |
|---|---|---|---|
| S0 | Ingestion metrics (§8.1) over the ingestion golden set and the full corpus. Two Docling configs: **P1** = fast table mode, **P2** = accurate table mode; OCR engine fixed (D10) | ingest set; full corpus for Auto metrics | **The parser config used downstream**, by cell accuracy (ties → faster). Otherwise characterizes parsing quality |
| S1 | C1–C5 × `R-dense`, plus §8.2 chunk metrics | dev | Top 2 chunkers by `recall@budget` |
| S2 | Top 2 chunkers × {`R-dense`, `R-bm25`, `R-hybrid`, `R-pc`}, each ± rerank | dev | Top 3 configurations by `recall@budget`, plus the C1 + `R-dense` baseline |
| S3 | Winner × {Flat, HNSW, IVF} | dev | No pick; recall vs latency curve only |
| S4 | Top 3 + baseline, with full generation (§8.4, §8.5) | **test** | Final reported numbers. Ranked by answer correctness |
| S5 | Load test of the baseline and the S4 winner (§9.4) | dev + test questions, repeated | Latency percentiles only |

At most 16 runs in S2. Every stage reports every configuration it ran, not just the winners.

### 9.4 Load test (S5)
*(Cut down 2026-11-04 to fit the Groq free tier of 200,000 tokens/day; ~2,000 tokens per request.)*
- **Configurations:** the **baseline** (C1 + `R-dense`) and the **S4 winner** only.
- **Volume:** 150 requests per configuration, cycling through all 120 questions in shuffled order (~300 requests, ~0.6M tokens, about 3 days).
- **Concurrency:** 1 only (latency without queueing). Record any 429 (rate-limited) responses as their own count rather than as latency.
- **Warm-up:** discard the first 10 requests of each configuration.
- **Reported:** p50 / p95 for TTFT, TTFS, retrieval latency and end-to-end latency, plus output tokens/sec and error rate. **No p99:** 140 requests give too few points in the tail.
- **Sessions:** S5 spans several days. Each day, run both configurations back to back in alternating order, and record each session's start time and requests per configuration.
- **Cost:** answers are not judged in S5, so it costs generation tokens only. The cost guard applies.
- Provider latency varies by time of day. Run every configuration's pass back to back in one session, alternating configuration order between passes, and record the start time.

### 9.5 Native streaming in the eval harness
The eval harness calls the Groq API with native streaming and timestamps each received chunk with a monotonic clock. TTFT, TTFS, inter-token latency and tokens/sec all come from these timestamps. Production `/query/stream` already streams natively (§4); the harness calls Groq directly so that retrieval time and provider time are measured separately and no local server sits in the path. The production endpoint is not changed (§3).

### 9.6 Statistics
- Every configuration answers the same questions, so comparisons are **paired**.
- For each ★ metric, use a paired bootstrap over per-question differences: 10,000 resamples, 95% CI.
- Call a configuration better only when the CI excludes 0. Otherwise report "no detectable difference at n = …".
- S4 compares 3 configurations to the baseline. Apply Holm correction across those 3 and say so in the report.
- Ingestion metrics (S0) are descriptive: report point estimates with bootstrap CIs over pages, and n pages per stratum. The S0 pick (P1 vs P2) uses a paired bootstrap over labeled tables.
- Non-★ metrics are reported as point estimates, labeled diagnostic.

### 9.7 CI gate
The corpus is public, so CI runs on **real public documents**, not synthetic ones.
- **CI fixtures:** 3–4 small documents from the corpus (e.g. one Loan Estimate, one Closing Disclosure, one short amendment, one simulated-scan page set), with their cached parser output committed, so CI never runs Docling. The 20-question smoke set is drawn from the **dev** split, covers every question type, and is already verified (no extra review).
- A GitHub Actions job on every PR runs **retrieval and chunk metrics only** on the smoke set. No LLM calls, no network beyond package install.
- It fails the build if `recall@budget`, `mrr@10` or entity recall drops more than 0.05 below `evals/baseline.json` for the production configuration, or if the embedder truncation rate rises above its baseline.
- **What the gate does and doesn't show:** it catches code regressions in chunking and retrieval. 20 dev questions are not a result. Never quote CI numbers as results.
- Updating `baseline.json` is a deliberate change made in its own commit, with the reason in the commit message.
- Ingestion metrics (S0), generation evals (S4) and the load test (S5) run locally only, since they need the full corpus, the Groq key, and time or money.
- The existing deploy workflow (`.github/workflows/deploy.yml`) targets EC2 hosts that no longer exist. Disable it in R0 so PRs and merges don't fail on it.

## 10. Requirements

**Functional**
- F1: Every chunker emits provenance (doc_id, page range, char offsets). A unit test checks offsets against the cleaned text for every chunker.
- F2: All retrievers share one interface: `retrieve(question, k) -> list[(chunk_id, score, rank)]`.
- F3: The QA validator fails if any `span_text` isn't found in the cleaned page text, if any listed `entity` isn't found in the evidence spans or reference answer, if any `superseded_values` entry isn't found in the session's documents, if a `qid` is duplicated, or if a split's type mix drifts more than ±1 question from proportional.
- F4: The ingestion-label validator fails if a table's cell grid is inconsistent (overlapping spans, missing cells), if a `reading_order` entry references a missing element, or if a bounding box falls outside the page.
- F5: The runner never re-parses a cached document, and never re-embeds a cached chunk set.
- F6: Any test-split run is recorded with a `test_run: true` tag. The S4 report states how many times the test split was run (target: once).
- F7: `report.py` generates the README results tables and charts from MLflow runs. Numbers are never typed in by hand.
- F8: Every metric in §8 has a unit test on a small hand-built fixture with a known answer (e.g. a 3-chunk ranking with known MRR, a 2×2 table with known cell accuracy, a string pair with known CER).
- F9 (changed): A test fails if git tracks any raw corpus PDF, cache directory, MLflow store, or `.env` file, and the fetch script verifies every downloaded document's sha256 against the manifest. `.gitignore` covers `data/`.
- F10 (changed): `report.py` may include example questions, answers and excerpts from the public corpus. Every excerpt cites its `doc_id` and source URL from the manifest.
- F11 (new): The fetch script is idempotent and rate-limited to the SEC fair-access limit, and reads its User-Agent from `SEC_USER_AGENT`; it refuses to run without it.

**Non-functional**
- N1: S0–S3 complete on a laptop CPU, with no GPU required.
- N2: Production behavior is unchanged unless R0 changes it. The default config must reproduce C1 + `R-dense` exactly, checked with a test that compares chunk IDs and text against the pre-refactor output on one fixture document.
- N3: The 122 existing tests keep passing (the 2 known failures are fixed or explicitly marked in R0). New deterministic code is tested with pytest. Judge-graded behavior is covered by the calibration in §8.6, not unit tests.
- N4: Secrets come only from environment variables. No AWS credentials are needed anywhere in the evals.
- N5: Timing uses a monotonic clock, measured in the client process. Network time is included and stated as such.

## 11. Pre-registered Hypotheses
Write these into the results before S0 runs, and report each as supported, not supported, or inconclusive. Don't add hypotheses after seeing results without labeling them post-hoc.
- **H1:** C4 (layout-aware) beats C1 on `recall@budget` for `table` questions.
- **H2:** `R-hybrid` beats `R-dense` on `lookup` questions that contain exact identifiers (rates, amounts, section numbers, defined terms).
- **H3:** Reranking improves `mrr@10` for every base retriever.
- **H4:** `R-pc` improves faithfulness on `explain` questions compared with the same chunker under `R-dense`.
- **H5:** The best configuration's $ per 1k queries is within 20% of the baseline's, because the context budget is fixed.
- **H6:** Simulated-scan pages have at least twice the CER of the digital pages they were rendered from.
- **H7:** Test questions from the lowest ingestion-score tercile have lower answer correctness than those from the highest tercile.
- **H8:** C1's embedder truncation rate is above 50%, and C2–C5's are below 5%.
- **H9 (new):** Under the baseline, amendment questions have the lowest `recall@budget` of any question subgroup, and a stale-answer rate above 20%.
- **H10 (new):** P2 (accurate table mode) beats P1 on cell accuracy for the table-heavy stratum.

## 12. Milestones

| # | Deliverable | Est. |
|---|---|---|
| R0 | Fix the retriever-cache eviction bug, with a regression test. Fix the `groq_model` code default. Disable the EC2 deploy workflow. Fix or mark the 2 failing tests. Measure MiniLM truncation: what fraction of current C1 chunks exceed the model's max sequence length (256 word-piece tokens for all-MiniLM-L6-v2; confirm from the loaded model's `max_seq_length`), and how much text that drops. Use `loan1.pdf`'s cached or re-parsed text for now; re-measure on the corpus in R2. Record the finding | 3h |
| R1 | **Corpus:** EDGAR deal search, candidate list for Kousik's approval, manifest, fetch/render script, scan simulator. **Parser:** interface, Docling adapter, parser cache. Ingestion labeling script and validator. Kousik labels 30 pages (5 from scratch). §8.1 metrics with F8 tests. `.gitignore` and the F9 test. **Run S0** | 12h agent + ~6.5h Kousik |
| R2 | Chunker interface with provenance; C1 refactored onto it (N2 test); C2–C5; §8.2 chunk metrics; tests | 10h |
| R3 | QA golden-set schema (with `superseded_values`), validator, drafting script with amendment pairing, dev/test split. Kousik verifies 120 questions | 5h agent + ~6h Kousik |
| R4 | §8.3 retrieval metrics, runner, chunk/embedding cache, local MLflow logging, report generator. **Run S1** | 7h |
| R5 | `R-bm25`, `R-hybrid`, `R-pc`, rerank; index-type study. **Run S2 and S3** | 8h |
| R6 | Generation harness with native streaming and §8.5 telemetry; citation prompt; judge with §8.4 metrics; calibration (Kousik labels 50). **Run S4 on test** | 8h agent + ~2h Kousik |
| R7 | Load test (§9.4). **Run S5.** Paired bootstrap + Holm; CI fixtures and smoke set from the public corpus; CI smoke gate and `baseline.json` | 7h |
| R8 | §8.7 analyses (quality/latency/cost chart, error attribution, confidence vs accuracy); hypothesis results; README results section generated by `report.py`; written rollout recommendation (does the winner fit a 1GB host? how to port production from Textract/S3 to Docling/local storage?) | 4h |

**Total: ~64h agent + ~14.5h Kousik.**

The data rules (D11) apply from R1 onward: `.gitignore` entries and the F9 test land in R1, before any document is downloaded.
**Minimum demonstrable slice:** R0–R4. That gives a public corpus of real credit agreements and loan forms, a measured picture of parsing quality, five chunkers with their own metrics, a span-labeled QA set, and S1 retrieval results with confidence intervals. S1 already answers whether layout-aware chunking helps, and H9 already shows whether amendments break retrieval.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Overfitting choices to the golden set | Pick on dev, report on test once (F6). Pre-registered hypotheses (§11) |
| Too many metrics: some will "win" by chance | One decision metric per stage (D3). CIs only on ★ metrics. Everything else is labeled diagnostic |
| QA set too small for per-type claims (~13–17 test questions per type) | Report per-type n beside every per-type number. Make overall claims only where CIs support them |
| Ingestion set too small for per-stratum claims (6–10 pages each) | Report n pages per stratum. Treat per-stratum results as descriptive |
| Pre-filled ingestion labels biased toward Docling | 5 pages labeled from scratch first (§5.4); report them separately |
| The PDF text layer is itself wrong or out of order | Text-layer sanity checks (§8.1). Numeric fidelity uses multisets, so order doesn't matter. Report excluded pages |
| LLM-drafted questions biased toward easy, extractive lookups | Fixed type quotas (§5.1). Kousik writes some `cross_doc` and `calculation` questions himself |
| Judge disagrees with humans | κ ≥ 0.6 gate (§8.6). Numeric exact match, refusal correctness and stale-answer rate don't depend on the judge |
| Chunk-size bias in `recall@k` | `recall@budget` is the decision metric (§8.3) |
| MiniLM truncation confounds the chunker comparison | R0 measures it. C2–C5 cap chunks at 256 MiniLM tokens. Report truncation next to each chunker's results (H8) |
| Provider latency varies by time and load | S5 runs configurations back to back in alternating order (§9.4). Report the run time. Latency claims are about this provider at this time, not absolute |
| Rate limits during the load test (free tier especially) | Concurrency kept low; 429s counted separately; S5 split across sessions if daily limits require it (§9.4) |
| Groq model retired mid-project (this has happened before) | Record the exact model ID per run. If retired, re-run S4 and S5 on the replacement and report both |
| Refactoring C1 changes production behavior | N2 equality test against pre-refactor output |
| Cross-encoder too heavy for a small host | Out of scope here. R8 measures memory and latency and recommends; no deployment |
| **Docling too slow on a laptop CPU for ~2,500 pages** (new) | Measure seconds per page on a sample first; parse once and cache; time cap in D6; shrink the corpus (fewer or shorter deals) before raising the cap |
| **Docling's confidence signals are thinner than Textract's** (new) | §8.1 uses what exists (OCR word confidence, page-level scores); H7 is computed separately for OCR'd and digital documents; if no usable digital-page score exists, H7 is tested on OCR'd documents only and says so |
| **Most EDGAR exhibits are HTML, not PDF** (new) | Prefer filings with PDFs; render the rest with fixed headless-Chromium settings; tag `source_format` and split S0 by it. Rendered PDFs are clean digital PDFs, which is stated in the report |
| **Simulated scans don't behave like real scans** (new) | Label them simulated everywhere; vary degradation levels; add real public scans if found (Open Question 1). No claim about real-scan accuracy without real scans |
| **The cleaner was written for Textract output** (new) | Over-cleaning rate and repair counts (§8.1) show its effect on Docling output; any cleaner change is a separate, versioned config, not a silent edit |
| **Amendment questions are ambiguous** (which value is "current"?) (new) | Each amendment question names the point in time ("after the Second Amendment"); `superseded_values` makes stale answers measurable |
| CI fixtures don't represent the full corpus | The CI gate is only a regression check (§9.7). All reported results come from local runs on the full corpus |

## 14. Decisions

These are fixed for the build. If one turns out to be wrong in practice, stop and raise it. Don't work around it.

**D1. Golden sets.** QA: 120 questions, 40 dev / 80 test, type quotas as in §5.1, corpus-part spread as in §5.1, split fixed before S1 and never reshuffled. Ingestion: 30 pages with the stratum quotas in §5.4, 5 of them labeled from scratch.

**D2. Evidence labeling.** By page and span text, with ≥ 50% character overlap counting as coverage (§8.3). Never by chunk ID.

**D3. Decision metrics.** Exactly one per stage:
- S0: cell accuracy (picks P1 or P2)
- S1 and S2: `recall@budget`
- S4: answer correctness, with faithfulness as a guard (a configuration whose faithfulness CI is entirely below the baseline's can't win)
- Ties (CI includes 0) go to the configuration with lower $ per 1k queries, then lower p95 end-to-end latency (S0: lower parse time)

The ★ metrics are headline numbers reported with CIs. All other metrics are diagnostic.

**D4. Models.**
- Parser: Docling, version pinned at R1 start; OCR engine chosen per D10.
- Embedder: all-MiniLM-L6-v2 for every configuration.
- Generator: `qwen/qwen3.8-27b` on Groq, the production model (decided 2026-11-04; `llama-3.3-70b-versatile` was retired). Record the ID the provider returns; if it is retired too, re-run S4/S5 on the replacement and report both.
- Judge: `openai/gpt-oss-120b` on Groq — a **different family** than the generator, so no model grades its own style. Confirmed live on 2026-11-04. It is a reasoning model: its hidden reasoning tokens are billed and count toward the daily limit, so the cost guard and token ledger must include them.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**D5. Citations.** The eval harness prompt asks the model to cite `[chunk_id]` after each claim. This is used for citation precision and recall, and to help the judge. Production prompts don't change in this initiative; R8 recommends whether to adopt it.

**D6. Parsing run (replaces v1's Textract re-extraction).** Parse the full corpus once per parser config (P1, P2) and cache. Before running, print the estimated CPU time from a 10-page sample; **if the total for both configs is over 8 hours, stop and ask.** If needed, run P2 on the ingestion set and table-heavy documents only, and record that.

**D7. Spend cap.** For LLM calls: $5 per runner invocation, $10 for the S5 load test, $40 total across the initiative. The runner tracks cumulative spend in a local ledger file and refuses to exceed the total without an explicit override flag. Spend is computed from published prices even on a free tier, so the ledger reflects real cost.

**D8. Where it runs.** All experiments run locally. No cloud services other than the Groq API (and EDGAR/CFPB downloads).

**D9. Build scope.** Build R0–R4, then stop for review before R5.

**D10. Choices left to the agent.** Decide these, then record each in the Decision Log at the end of this doc with a one-line reason:
- the BM25 library (must be in-process and pip-installable)
- the TEDS implementation and the PDF text-layer library
- the Docling OCR engine and its settings, and the Docling-label → §5.4-class mapping
- the HTML→PDF rendering settings and the scan-simulation degradation levels
- text normalization rules for span matching, CER, entity matching and numeric parsing (case, whitespace, punctuation, currency symbols, thousands separators, percent and basis-point forms)
- the HNSW and IVF parameters for the index study
- the parent window size for `R-pc` when no C4 section is available
- the judge rubric wording (versioned; changes after calibration get a new version)
- the exact MLflow metric names, as long as they map one-to-one onto §8
- how the 20-question CI subset is chosen (it must include every question type)
- the ingestion labeling interface (a script plus a JSON editor is enough; no web app required)

**D11. Data (replaces v1's private-data rules).** The corpus is public.
- **Committed:** code, schemas, validators, configs, the corpus manifest with source URLs and checksums, the fetch and scan-simulation scripts, the QA golden set, the ingestion labels and their 30 page images, CI fixtures with cached parser output, `baseline.json`, and aggregate results. Final S4 per-question results may be committed for transparency.
- **Not committed (gitignored under `data/`):** raw downloaded PDFs (re-created by the fetch script), parser and chunk/embedding caches, the local MLflow store, the spend ledger.
- **Attribution:** every document keeps its source URL in the manifest, and reports cite it. CFPB forms are US government works; EDGAR filings are public records. Record both facts in the decision log after checking each source's terms.
- **Backup:** golden sets and labels live in git, so no separate private store is needed.

## 15. Open Questions (for Kousik)
1. **Real scans:** are any real scanned public loan documents worth adding (e.g. older paper filings)? Without them, H6 and the scanned stratum rest on simulated scans only.
2. ~~Multi-column pages?~~ **Resolved 2026-11-04: yes,** the complex-layout stratum may come from consumer forms.
3. ~~Groq tier?~~ **Resolved 2026-11-04: free, 200,000 tokens/day.** S4 needs several days; S5 as written (~8M tokens) does not fit. See Open Question 6.
4. **Deal list:** approve the 12 EDGAR deals before download (R1).
5. ~~Related projects?~~ **Resolved 2026-11-04:** placeholder PII module (`app/privacy/pii.py`) and Airflow DAG (`orchestration/`) built in R0, inspired by those projects, **not connected** to the pipeline or the evals.
6. ~~Generator model and token budget?~~ **Resolved 2026-11-04:** generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b` (D4). S5 load-tests the baseline and the winner only (§9.4).

Resolved in v1 and still valid: cost caps for LLM calls (D7); no score thresholds on non-dense retrievers (§7). Superseded by v2: v1 Open Question 1 (public text — now public by design, D11); v1 Open Question 4 (corpus scans — now Open Questions 1–2 here).

## 16. What This Enables on the CV (rules)
Numbers can be quoted only after the stage that produces them has run on its final data: S0 for parsing, S4 on the test split for quality, S5 for latency percentiles. Each needs its n and, for ★ metrics, its CI. For example: "layout-aware chunking raised recall at a fixed 1,200-word context budget from X to Y on real SEC credit agreements (n = 80, 95% CI …)", "the baseline gave stale pre-amendment values in X% of amendment questions; the winning configuration cut that to Y%", or "p95 time-to-first-token of X ms over 140 requests". Hypotheses that weren't supported are reported as such; they're still good interview material. No claim may rest on the dev split, a diagnostic metric alone, or the CI smoke set. Claims about scanned documents say "simulated scans" unless real scans were added. Because the corpus is public, example questions and answers from it may be shown in interviews and the README, with their source cited.

## Decision Log
*(The agent adds entries here for D10 choices: date, choice, one-line reason.)*

| Date | Choice | Reason |
|---|---|---|
| 2026-11-04 | **EDGAR deal discovery:** full-text search for amendment titles in EX-10 exhibits (2019–2025); amendments grouped by the date of the agreement they amend; original = same company's EX-10 filed within 100 days of that date, titled "Credit Agreement" and showing that date in its opening text (`evals/corpus/edgar_discover.py`) | Ties each amendment to its exact agreement without hand-checking; the date check rejected 5 of 37 first-pass matches |
| 2026-11-04 | **Duplicate / ambiguous amendments:** same amendment number + same own date = one document filed twice; keep the copy with fewest `[***]` redactions, then the longest. Same number + different dates = two agreements share a date, so the deal is dropped | Companies refile amendments (8-K then 10-Q; redacted then full). FirstEnergy and Omega were dropped as ambiguous |
| 2026-11-04 | **Rendering:** headless Chrome 156, US Letter, 0.5in margins, no header/footer; scripts and images stripped; a Content-Security-Policy blocks all network loads | No new dependency; nothing is fetched from EDGAR without our User-Agent. 356 images removed (page backgrounds, logos, signatures); PDFs keep 95.7–100% of the HTML's words (median 98%) |
| 2026-11-04 | **Page fidelity:** render at zoom 1.0 / 0.9 / 0.8 / 0.7, keeping the first whose page count ≤ the source's page breaks + 1; zoom recorded per document in the manifest | At a fixed 0.75in/1.0 setting, original pages spilled onto extra PDF pages (deal_12: 297 PDF pages for 176 originals), which breaks page-level evidence labels. Result: 23 docs at 1.0, 14 at 0.9, 3 at 0.8, 1 at 0.7 |
| 2026-11-04 | **Checksums on source HTML, not PDFs** | Chrome embeds timestamps, so PDFs are not byte-stable; PDFs are identified by page count |
| 2026-11-04 | **CFPB forms imported by hand, not scraped:** consumerfinance.gov returns 403 to scripts; the 11 PDFs are saved manually into `data/corpus/inbox/cfpb/` and imported with sha256 frozen in the manifest (`evals/corpus/cfpb.py`) | Don't work around a site's bot protection; checksums still make every copy verifiable |
| 2026-11-04 | **CFPB sessions by content:** purchase (H-24(B) + H-25(B): 456 Somewhere Ave, $162,000), refinance (H-24(D) + H-25(E)/(F)/(G): 123 Anywhere St, $150,000); H-24(C)/(E)/(F) and H-25(C)/(D) each a separate session | C/E/F reuse the purchase borrower with different loan amounts and C/D are stand-alone page-3 excerpts; grouping them would put contradicting figures in one session. 11 forms, 40 pages, 7 sessions |
| 2026-11-04 | **Scan degradation levels** (grayscale, then skew, paper tone, blur, noise, JPEG): light = 300 dpi, blur 0.3, noise σ4, skew ±0.3°, JPEG q85; medium = 200 dpi, 0.7, σ10, ±1°, q60; heavy = 150 dpi, 1.1, σ18, ±2°, q35 (`evals/corpus/scans.py`) | Spans clean office scans to poor copies while staying human-readable (checked visually); levels assigned round-robin within each stratum (14 light / 14 medium / 12 heavy) |
| 2026-11-04 | **Scan page selection:** seed 20261104; quotas edgar_prose 12, edgar_table 8, edgar_amendment 8, consumer 12; ≥ 80 words; signature pages excluded (≥ 4 By:/Name:/Title: markers); ≤ 2 pages per document per stratum, ≤ 3 per document | A first cap of 2 per document left the table quota short (6/8): detectable tables sit in 8 of 41 EDGAR docs (33 pages). Signature pages (3 in the first draw) are mostly blank. Result: 40 pages from 24 documents; rebuilds are byte-identical |
| 2026-11-04 | **Table detection for selection = PyMuPDF `find_tables` (ruled tables, ≥ 3 rows)** | Cheap and deterministic; unruled tables count as prose here. The ingestion golden set labels tables by hand, so this only affects which pages get scanned |
| 2026-11-04 | **Docling in its own venv** (`venv-docling`, pinned in `requirements-docling.txt`): Docling 2.147.0 would upgrade numpy 1.26→2.5, torch, httpx, pydantic-settings in the app venv | Protects the app and its 169 tests; the parse cache (JSON) is the only interface between the two envs |
| 2026-11-04 | **Device: Apple GPU (MPS)** on the M3 | Kousik's request. Measured on a 10-page sample: MPS 1.54 / 2.17 s/page (fast / accurate tables) vs CPU 1.74 / 2.33, about 10% faster; most time is PDF parsing and OCR, not the GPU models |
| 2026-11-04 | **OCR engine: Tesseract 5.5.1 (CLI)** | Compared on 5 simulated scans: Apple Vision (ocrmac) read best (80–99% word recall) but gives only 1.0/0.5 confidence, useless for §8.1 confidence metrics and H7; EasyOCR gave continuous confidence but failed on heavy scans (2–41% recall); Tesseract: 63–99% recall with continuous per-word confidence that tracks page difficulty. Apple Vision's numbers on the 40 scans are reported in S0 as a diagnostic |
| 2026-11-04 | **Key-value pairs:** Docling's `extract_form_fields` only reads fillable PDF form fields; the CFPB samples are flat, so Docling returns 0 key-value pairs | Kept as is: the §8.1 key-value metrics will report Docling's real (near-zero) performance as a finding, per scope |
| 2026-11-04 | **Parse cache = Docling's lossless document JSON + confidence report + OCR cells,** one file per document per config, keyed by hash(doc sha256, Docling version, config); resumable runner with the D6 8-hour projection stop (`evals/parsers/`) | Metrics and chunkers read everything from the cache; the app venv never imports Docling |
| 2026-11-04 | **Full parse run:** P1 and P2 over 92 documents (1,916 pages each) in 19.5 + 20 minutes on the M3 GPU, 0 failures | Far under the 8-hour cap (D6); the 10-page sample overestimated because it was 30% scans |
| 2026-11-04 | **CFPB cover sheets excluded from scan and golden-set selection** (page 1 of all 11 sample PDFs is a cover, not the form); covers stay in the corpus | 3 of the first 6 complex-layout picks and 1 simulated scan were covers. Scans rebuilt: 8 changed, 32 byte-identical; only the 8 were re-parsed |
| 2026-11-04 | **Ingestion golden-set selection** (seed 20261105, Docling-independent): digital_prose 10 (EDGAR, no detected table), table_heavy 8 (5 EDGAR + 3 CFPB with a PyMuPDF-detected table), scanned 6 (2 per level), complex_layout 6 (other CFPB form pages); ≥ 80 words, no signature or cover pages, ≤ 1 page per doc per stratum, ≤ 2 per CFPB form, scan source pages excluded; 5 from scratch (2 prose, 1 per other stratum) | Selecting without Docling keeps the set from favouring pages Docling handles well. CFPB form pages count as complex layout with or without ruled tables (nearly all have them), after the consumer table quota fills |
| 2026-11-04 | **Label conventions:** PDF points, top-left origin; Docling labels mapped title→title, section_header→section_header, text/paragraph/footnote/caption→text, list_item→list, table/document_index→table, page_header/page_footer→same, a footer that is only a page number→page_number, everything else→other; Docling's missing empty table cells filled as empty cells so every grid is complete | Matches the §5.4 classes; IoU ≥ 0.5 matching tolerates rough boxes. Prefill is a starting point; Kousik's corrections are the truth |
| 2026-11-04 | **Labeling interface:** CLI (`evals/golden/ingest/labeling.py`) + JSON files + overlay images with a 50-point grid; guide in `evals/golden/ingest/README.md` | D10 allows a script + JSON editor; overlays make box coordinates readable without a web app |
| 2026-11-04 | **Local labelling app added** (`evals/golden/ingest/app/`: stdlib HTTP server on 127.0.0.1 + one offline HTML page): drag/resize/draw boxes, type keys, table grid editor with spans and paste-to-fill, form fields, reading order, live validation, undo; refuses "verified" while the validator reports problems | Kousik found JSON + coordinates hard to use; the app writes the same files, so the CLI, validator and metrics are unchanged. Tested end to end in headless Chrome |
| 2026-11-04 | **Normalization (D10):** text = NFKC, straight quotes, dashes → "-", whitespace collapsed, case kept (CER); matching = text + lower case + punctuation other than `. , % $ / -` dropped (cells, keys, headers/footers); numbers = `$` and thousands separators (any grouping) dropped, `%`/"percent" → `%`, `(1,000)` → `-1000`, dates kept as written, a hyphen after a digit/letter is a range not a minus, digits glued to letters are identifiers (`evals/ingest_metrics/normalize.py`) | Deterministic, tested on known inputs; keeps rates, amounts, dates and section numbers comparable across parsers |
| 2026-11-04 | **Cell accuracy:** counts only non-empty labelled cells (an empty string "appears" in anything); a missed table scores 0 on all its cells; "appears in" = normalized containment, per §8.1 | Prevents empty cells inflating the score |
| 2026-11-04 | **TEDS:** standard PubTabNet costs (structure mismatch 1; same-shaped cells cost normalized text edit distance) with the `apted` tree-edit library; missed table = 0 | Scope asks for an existing open-source implementation; `apted` is the library the reference TEDS code uses |
| 2026-11-04 | **Split-table rate not computed:** per-page labels can't say a table continues on the next page | Known gap; would need cross-page labels |
| 2026-11-04 | **Reading order:** Kendall τ over body elements matched by IoU (any class); headers/footers/page numbers excluded | Furniture placement isn't reading order |
| 2026-11-04 | **Docling list numbers** ("3.", "(a)") live in a separate `marker` field; the shared reader now prepends them | Without it, SEC numeric fidelity read 92.2% instead of 95.2%; section numbers are content |
| 2026-11-04 | **Order-free word recall added as a diagnostic next to CER** | CER against the text layer mixes character errors with reading order: CFPB pages show 31% CER but 99.7% word recall; scans showed lower CER than their own digital source pages |
| 2026-11-04 | **H6 paired per scan with its own source page** (not all digital pages) | Matches the hypothesis wording; the first version compared against an average inflated by CFPB reading order |
| 2026-11-04 | **P1V diagnostic** (P1 + Apple Vision OCR) parsed on the 40 scans only | Reports what choosing Tesseract costs: heavy scans CER 81% (Tesseract) vs 5.5% (Vision); numbers right 41% vs 73% |
| 2026-11-06 | **Downstream OCR = Apple Vision; confidence = Tesseract** (Kousik's decision). Simulated scans feed chunking/retrieval/generation from P1V/P2V (Apple Vision); digital docs use P1/P2. Confidence metrics (§8.1) and H7 use the Tesseract run. Rule in `evals/parsers/config.downstream_config` | Heavy scans: Vision 73% of numbers right vs Tesseract 41%; Tesseract's confidence is the informative one. Mirrors a real design (cheap OCR pass as an upload quality check). Caveat for H7: the confidence comes from a different engine than the text used for answers; state this in the H7 result |
| 2026-11-04 | **Approved deals (12):** Lincoln Educational, EQT, Raymond James, IFF, SigmaTron, SWK, Harmonic, PureCycle, United Therapeutics, RPM, TrueBlue, Tactile Systems (`evals/corpus/edgar_selection.json`) | Approved by Kousik. 23 industries in the candidate pool; picks favour smaller agreements and industry spread. Corpus: 41 documents, 1,836 pages |

| 2026-11-06 | **S0 pre-registration** in `docs/20_s0_results.md`, committed before the run. H6 test: P1/P2 (Tesseract) only, Vision descriptive; disclosed as already seen in dry runs. **H10 test:** paired bootstrap of P2 − P1 cell accuracy over `table_heavy` tables; CI above 0 = supported, below = not supported, else inconclusive (`run_s0.h10`) | Scope §11: hypotheses written into the results before S0; H10 had no computation yet |
