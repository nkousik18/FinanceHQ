# LoanDoc Retrieval & Evals Initiative — Scope

**Status:** Draft · **Owner:** Kousik Nandury · **Date:** 2026-11-01 (metrics catalog expanded 2026-11-03)
**Repo:** github.com/nkousik18/FinanceHQ (internal name FinanceHQ, public name LoanDoc)
**Naming:** This is the "Retrieval & Evals" initiative. Milestones are numbered **R0–R8** so they don't collide with any existing phase or milestone names in the repo's roadmap. If the repo has a roadmap file, add a section for this initiative there rather than renumbering anything.

## 1. Problem
LoanDoc is a working RAG system: Textract extraction, one sentence-window chunker, dense FAISS search, Groq generation. Its evaluation is weak:
- 7 questions over one synthetic document
- "Groundedness" is a keyword match, and "relevance" is word overlap between question and answer
- **Retrieval is never measured.** A wrong answer can't be traced to a bad chunk or a bad model
- **Parsing is never measured.** Nobody knows how accurately Textract reads the tables, forms and layout of these documents, or whether its confidence scores mean anything
- **Latency is measured only end to end,** and the streaming endpoint simulates streaming, so time-to-first-token has never been measured

AI engineering postings ask for grounded, cited answers (36 of 166 saved postings), vector search (33), and eval sets and pipelines (28), along with measured cost and latency. Few name a specific retrieval technique (hybrid 5, rerank 3, graph 4, hierarchical 1). A plain RAG app with no measurement doesn't meet that bar.

## 2. Goal
Build an offline evaluation pipeline that measures every stage of LoanDoc:
- **ingestion and parsing** against a labeled page set
- **chunking**, with metrics computed from the chunks themselves
- **retrieval** against a span-labeled question set
- **generation**: quality, latency and cost

Each comparison comes with statistical confidence intervals. Use the pipeline to run a staged comparison of 5 chunkers and 5 retrieval strategies, plus a load test for latency. Then report which configuration wins on this data, by how much, at what cost and latency, and which stage causes each remaining wrong answer.

The experiment and the eval pipeline are one system. The comparison is the first thing the pipeline is used for.

## 3. Non-goals
- **No embedding-model comparison.** all-MiniLM-L6-v2 stays fixed as the control variable (see R0 for its truncation check). Swapping embedders is a possible follow-up.
- **No replacing Textract.** Ingestion is measured, not changed. The one exception is adding the `LAYOUT` feature (D6). Comparing other parsers is a possible follow-up, and §8.1 makes it easy later.
- **No graph RAG.** Loan documents are mostly independent files, so the cost is high for little expected gain.
- **No production deployment of the winning configuration.** The EC2 t2.micro hosts have 1GB RAM, which may not fit BM25 plus a cross-encoder. R8 produces a written rollout recommendation only.
- **No online A/B testing.** There is no live traffic. The existing epsilon-greedy prompt selector stays as it is and is not part of this work.
- **No LLM fine-tuning, no self-hosted model serving.** Those belong to the separate inference module.
- **No new vector database.** FAISS stays in-process. Every retriever must run inside the same process with no new services.
- **No UI changes,** and no change to the production streaming endpoint. Native streaming is used inside the eval harness only (§9.5).

## 4. Current State (from the project profile; verify against the code before relying on it)
- **Extraction:** AWS Textract `AnalyzeDocument` with TABLES + FORMS, sync for 1-page docs, async for 2+. Output objects include `ExtractionResult`, `PageResult` and `Table` (`Table.to_markdown()`).
- **Cleaning:** 7 ordered fixes: number-space repair, repeated header/footer removal, empty-checkbox artifact removal, duplicate-line removal, whitespace normalization, hyphenation repair, smart-quote/em-dash normalization.
- **Validation:** per-block Textract confidence, required fields, page count. `WARNING` logs and continues; `ERROR` fails the document.
- **Chunking:** sentence window via `nltk.sent_tokenize`, ~400 tokens (tiktoken `cl100k_base`), 60-token overlap, 50-token minimum. `chunk_id = "{session_id}-{doc_id}-{index}"`.
- **Embedding:** sentence-transformers `all-MiniLM-L6-v2`, 384-dim, CPU.
- **Index:** FAISS `IndexFlatIP` on normalized vectors. Score threshold 0.25, default `top_k=5`.
- **Context assembly:** 1,200-word budget, trims lowest-ranked chunks first.
- **Generation:** Groq `llama-3.3-70b-versatile`, temperature 0.1, max_tokens 1024. Five intent templates × two variants.
- **Streaming:** the SSE endpoint **simulates** streaming, emitting the finished answer word by word at 30 ms per word. Its timings are not real token timings.
- **Tracking:** self-hosted MLflow, artifacts in S3.
- **Corpus:** ~60 loan PDFs in varied formats, about 10–11 pages each (600+ pages).
- **Tests:** ~122 pytest tests, external dependencies mocked.
- **Known bug:** the retriever's session cache is not evicted when a second document is added to a session. Queries keep hitting the old index until restart.

## 5. Golden Sets
There are two:
- **QA golden set** (§5.1–5.3): questions with span-labeled evidence. Used for retrieval and generation metrics.
- **Ingestion golden set** (§5.4): hand-labeled pages. Used for parsing metrics.

### 5.1 QA set: size and mix
120 questions in total, split before any experiment runs:
- **dev (40):** used to pick winners at each stage
- **test (80):** run once at the end (stage S4) and used for every reported number

| Type | Count | What it tests |
|---|---|---|
| `lookup` | 25 | A single fact in running text (rate, tenure, lender name) |
| `table` | 25 | A value that only exists inside a table (fee schedule, amortization row) |
| `calculation` | 20 | Arithmetic over document values (total paid, remaining principal) |
| `cross_doc` | 15 | Needs evidence from 2+ documents in the same session |
| `explain` | 15 | A clause or condition in prose (prepayment penalty terms) |
| `unanswerable` | 20 | Plausible questions the documents don't answer |

The dev/test split keeps the type mix proportional.

### 5.2 QA set: schema (JSONL, one question per line)
```json
{
  "qid": "q0042",
  "split": "test",
  "type": "table",
  "question": "What is the late payment fee?",
  "session_docs": ["doc_017"],
  "evidence": [
    {"doc_id": "doc_017", "page": 4, "span_text": "Late payment charge: 2% per month on overdue amount"}
  ],
  "reference_answer": "2% per month on the overdue amount.",
  "key_points": ["2% per month", "applies to the overdue amount"],
  "entities": ["2%", "per month"],
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
- `span_text` must appear in the **cleaned** page text. A validator checks this (§10, F3).
- `key_points` are the facts a complete answer must contain (used for answer completeness).
- `entities` are the amounts, rates, dates and party names in the reference answer (used for entity recall). The drafting script pre-fills them; Kousik verifies them with the question.
- `unanswerable` questions have an empty `evidence` list.

### 5.3 QA set: how it's built
1. The agent writes a drafting script. It samples pages per document and asks an LLM for candidate questions by type, each with a proposed answer, evidence span, key points and entities.
2. The script auto-rejects any candidate whose span is not found verbatim in the cleaned text.
3. Kousik verifies every question that goes into the set: keep, edit or reject. **Nothing enters the set without his verification.** About 6 hours of his time.
4. The script assigns `qid`s and the dev/test split.

### 5.4 Ingestion golden set
**30 pages**, sampled from the corpus to fill these quotas:

| Stratum | Pages | Why |
|---|---|---|
| Digital, prose and forms | 10 | The common case |
| Table-heavy (fee schedules, amortization tables) | 8 | Tables are where loan answers live |
| Scanned or low quality | 6 | Where text accuracy and confidence matter most |
| Multi-column or complex layout | 6 | Where reading order and layout classification break |

If the corpus doesn't have 6 scanned or 6 multi-column pages, take what exists, fill the rest from the table-heavy stratum, and record the actual counts (Open Question 4).

**Labels per page (JSON, one file per page):**
- `page_id` (`doc_id` + page number), `stratum`
- `reference_text`: for scanned pages only. Digital pages use the PDF's embedded text layer as the reference instead (§8.1).
- `tables`: bounding box, cell grid (row, column, row span, column span, text), header row indices
- `key_values`: key text and value text
- `layout_elements`: bounding box and class (title, section header, text, list, table, page header, page footer, page number)
- `reading_order`: the layout element IDs in correct reading order
- `header_footer_lines`: lines that are page headers or footers

**How it's built:**
1. The agent writes a labeling script that **pre-fills** each page's labels from Textract's output and renders the page image beside them.
2. Kousik corrects every label. About 6 hours of his time.
3. To measure the bias pre-filling introduces (labelers tend to accept what's already there), Kousik labels **5 of the 30 pages from scratch** first, without seeing Textract's output. Report Textract's scores on those 5 separately from the other 25.

## 6. Chunkers
Every chunker implements one interface and emits chunks with **provenance**: `doc_id`, `page_start`, `page_end`, and character offsets into the cleaned page text. Retrieval metrics depend on provenance, so a chunker without it can't be evaluated.

| ID | Chunker | Rules |
|---|---|---|
| `C1` | Sentence window (baseline) | Current production behavior: ~400 tokens, 60 overlap, 50 min |
| `C2` | Sentence window, small | Same algorithm at ~200 tokens, 40 overlap. Tests chunk size alone, and fits inside MiniLM's input limit (R0) |
| `C3` | Fixed-size tokens | 256 tokens, 32 overlap, no sentence awareness. Simplest comparison point |
| `C4` | **Layout-aware** | Uses Textract's `LAYOUT` feature (requires re-extraction, see D6). Rules below |
| `C5` | Semantic | Splits where the embedding similarity between adjacent sentences drops below the 95th-percentile distance threshold for that document. Same min/max token bounds as C2 |

**C4 layout-aware rules:**
- Section boundaries come from `LAYOUT_TITLE` and `LAYOUT_SECTION_HEADER` blocks.
- Each chunk is prefixed with its heading path, e.g. `"Loan Agreement > Repayment > Late Charges: "`. The prefix is embedded, and it counts toward the token budget.
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

**Index types (separate, small study):** compare `IndexFlatIP` against `IndexHNSWFlat` and `IndexIVFFlat` on the winning configuration. Report recall against Flat and query latency. At ~600 pages, Flat is expected to stay fast enough; the point is the curve, not a winner.

**The 0.25 score threshold:** apply it only to `R-dense`. Scores from BM25, RRF and the reranker are on different scales, so the threshold doesn't transfer to them. Unanswerable questions are scored at the generation stage instead (§8.4).

## 8. Metrics

**How to read these tables:**
- **Source** says what a metric needs:
  - **Auto:** computed from data that already exists, no labels
  - **QA:** the QA golden set (§5.1)
  - **Ingest:** the ingestion golden set (§5.4)
  - **Judge:** the LLM judge, valid only after calibration (§8.6)
- **★ marks a headline metric.** Headline metrics are reported with confidence intervals. Every other metric is logged for diagnosis, without CIs and without being used to pick anything.
- **Only one metric picks the winner at each stage** (D3). With dozens of metrics, some will show a "win" by chance alone. A fixed decision metric and the pre-registered hypotheses (§11) guard against that.

### 8.1 Ingestion and parsing

**Textract confidence (Auto).** Textract gives every block a 0–100 confidence.

| Metric | Definition |
|---|---|
| Mean / p10 / min confidence | Per block type (WORD, LINE, TABLE, CELL, KEY_VALUE_SET, each LAYOUT_* type), reported per page and per document |
| Low-confidence word rate | % of words below 90, and % below 80 |
| Low-confidence page count | Pages whose p10 word confidence is below 80 |
| Document ingestion score | % of a document's words with confidence ≥ 90. Used in the analysis in §8.7 |

**Text accuracy.**
The reference text is the PDF's embedded text layer for digital pages (extracted with PyMuPDF or pdfplumber; Auto), and `reference_text` for scanned pages (Ingest). Use the text layer only on pages where it has at least 50 words and no more than 5% non-printable characters. Record each excluded page and the reason.

| Metric | Definition |
|---|---|
| ★ CER | Character edit distance between Textract's text and the reference, divided by reference length, after normalization (D10). Computed per page; reported as median and p90 across pages |
| WER | The same, counted in words |
| ★ Numeric fidelity | % of numeric tokens in the reference (amounts, rates, dates, percentages) that appear exactly in the cleaned output. Compared as multisets, so reading order doesn't affect it |
| Cleaning repair counts | How many times each of the 7 cleaning fixes fired, per document |
| Characters removed by cleaning | % of extracted characters the cleaner removed |
| Over-cleaning rate | Pages where CER after cleaning is worse than before cleaning |

**Tables (Ingest).**

| Metric | Definition |
|---|---|
| Table detection precision / recall / F1 | A predicted table matches a labeled table when their boxes overlap with IoU ≥ 0.5 |
| ★ Cell accuracy | % of labeled cells whose normalized text appears in the matched predicted cell at the same row and column |
| TEDS | Tree-edit-distance similarity between predicted and labeled table structure (0–1). Captures merged cells and row/column structure. Use an existing open-source implementation |
| Header-row accuracy | % of tables whose header rows were identified correctly |
| Split-table rate | % of labeled tables that span pages and come out as separate unlinked fragments |

**Forms / key-value pairs (Ingest).**

| Metric | Definition |
|---|---|
| Key-value precision / recall / F1 | Pairs matched on normalized key text |
| Value exact-match rate | Matched keys whose normalized value is exactly right |

**Layout (Ingest).**

| Metric | Definition |
|---|---|
| Element classification F1 | Per class and macro-averaged. A predicted element matches a labeled one at IoU ≥ 0.5 |
| ★ Section-header recall | % of labeled section headers detected as `LAYOUT_SECTION_HEADER` or `LAYOUT_TITLE`. C4 depends on this |
| Reading-order accuracy | Kendall's τ between predicted and labeled element order, per page. Report multi-column pages separately |
| Header/footer removal precision and recall | Lines the cleaner removed as headers/footers vs `header_footer_lines` |

**Throughput and reliability (Auto).**
- Pages per minute; stage latency p50/p95 for upload, extraction, cleaning, validation, chunking and indexing
- Textract $ per page, by feature set
- Async job wait time
- Failure and retry rate by stage
- Validation warnings and errors per document

### 8.2 Chunking (Auto unless noted; computed per chunker with no queries)

| Metric | Definition |
|---|---|
| Chunk count; size distribution | p5 / p50 / p95 / max tokens per chunk, in both tiktoken and MiniLM tokens |
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
| `context_words` | Auto | Average words passed to the generator |
| Retrieval latency p50 / p95 | Auto | Total, and per component: query embedding, vector search, BM25, fusion, rerank |
| Recall vs exact search | Auto | For HNSW and IVF: overlap of their top k with Flat's top k |

Unanswerable questions are excluded from all coverage-based metrics. Break every metric down **by question type** as well as overall.

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
| ★ $ per query, $ per 1k queries | Auto | Using the provider's published prices for the exact model, recorded with the run |
| Input-token vs output-token cost | Auto | The split of each query's cost. Input tokens are billed in full when the request starts, so there is no separate "cost to first token"; this split shows whether context size or answer length is driving spend |
| ★ Cost per correct answer | QA + Judge | Total $ ÷ (sum of answer-correctness scores) |
| Error and timeout rate | Auto | Failed or timed-out generation calls |

**All timings are measured client-side with native provider streaming** (§9.5). Latency **percentiles** (p50 / p95 / p99) are reported from the load test (S5), not from the 80 test questions: p95 of 80 rests on about 4 data points, and p99 on less than one. S4 reports latency means and medians only.

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
  4. **Generate:** the evidence was in context and the answer was still wrong

  Report counts and percentages per configuration.
- **Ingestion confidence vs answer accuracy:** bucket test questions by their source document's ingestion score (terciles) and compare answer correctness across buckets. If low-confidence documents fail more often, that supports flagging such documents at upload time. Report it as exploratory, with n per bucket.

## 9. Pipeline

### 9.1 Layout
The agent picks paths that follow repo conventions. Logically:
- `evals/golden/qa/` — validator, drafting script and schema are committed. The QA golden set itself is local-only and private (D11)
- `evals/golden/ingest/` — labeling script, validator and schema are committed. Page labels and page images are local-only and private (D11)
- `evals/fixtures/synthetic/` — synthetic documents, their extraction output and their 20-question smoke set, committed for CI (§9.7)
- `evals/ingest_metrics/` — the §8.1 metrics
- `evals/chunkers/`, `evals/retrievers/` — implementations behind shared interfaces. Production code should import the same chunker and retriever classes, rather than keeping a copy for evals
- `evals/configs/` — YAML, one file per stage
- `evals/run.py` — CLI runner
- `evals/loadtest.py` — S5 load test
- `evals/report.py` — generates the results tables and charts
- `evals/baseline.json` — stored smoke-subset scores for the CI gate

### 9.2 Runner
```
python -m evals.run --config evals/configs/s0_ingestion.yaml
python -m evals.run --config evals/configs/s1_chunkers.yaml --split dev
python -m evals.run --config evals/configs/s4_generation.yaml --split test --confirm-cost
python -m evals.loadtest --config evals/configs/s5_load.yaml --confirm-cost
```
- **Config file:** lists chunker × retriever × rerank × top_k combinations explicitly, never as a Cartesian product generated automatically.
- **Extraction cache:** Textract output stored once per document and extraction-feature set. Experiments never call Textract.
- **Chunk/embedding cache:** keyed by `hash(chunker_id, chunker_params, extraction_version, cleaning_version, embedder)`. Re-running a stage re-embeds nothing that hasn't changed.
- **MLflow:** one run per configuration, in a new experiment, `loandoc_retrieval_evals`. S0 and S5 use their own experiments, `loandoc_ingestion_evals` and `loandoc_load_tests`.
  - Params: every config field, git SHA, golden-set versions, model names, judge rubric version.
  - Metrics: everything in §8 that applies to the stage.
  - Artifacts: per-question (or per-page, for S0) results JSONL. For QA stages: qid, retrieved chunk IDs with ranks and scores, covered spans, entities found, answer, citations, judge grades with rationales, tokens, cost, and every timing in §8.5.
- **Cost guard:** stages that call an LLM print an estimated $ cost (questions × configurations × average tokens × price) and refuse to run without `--confirm-cost`. Caps in D7.
- **Determinism:** temperature 0 for generation and the judge in eval runs. Record model IDs exactly as the provider returns them.

### 9.3 Experiment stages
| Stage | Runs | Split | Picks |
|---|---|---|---|
| S0 | Ingestion metrics (§8.1) over the ingestion golden set and the full corpus. Textract features: TABLES + FORMS vs TABLES + FORMS + LAYOUT | ingest set; full corpus for Auto metrics | No pick; characterizes parsing quality |
| S1 | C1–C5 × `R-dense`, plus §8.2 chunk metrics | dev | Top 2 chunkers by `recall@budget` |
| S2 | Top 2 chunkers × {`R-dense`, `R-bm25`, `R-hybrid`, `R-pc`}, each ± rerank | dev | Top 3 configurations by `recall@budget`, plus the C1 + `R-dense` baseline |
| S3 | Winner × {Flat, HNSW, IVF} | dev | No pick; recall vs latency curve only |
| S4 | Top 3 + baseline, with full generation (§8.4, §8.5) | **test** | Final reported numbers. Ranked by answer correctness |
| S5 | Load test of the S4 configurations (§9.4) | dev + test questions, repeated | Latency percentiles only |

At most 16 runs in S2. Every stage reports every configuration it ran, not just the winners.

### 9.4 Load test (S5)
- **Volume:** 500 requests per configuration, cycling through all 120 questions in shuffled order.
- **Concurrency:** two passes. Concurrency 1 measures latency without queueing. Concurrency 4 measures latency under light contention. Report both. Stay within the provider's rate limits, and record any 429 (rate-limited) responses as their own count rather than as latency.
- **Warm-up:** discard the first 20 requests of each pass.
- **Reported:** p50 / p95 / p99 for TTFT, TTFS, retrieval latency and end-to-end latency, plus output tokens/sec and error rate.
- **Cost:** answers are not judged in S5, so it costs generation tokens only. The cost guard applies.
- Provider latency varies by time of day. Run every configuration's pass back to back in one session, alternating configuration order between passes, and record the start time.

### 9.5 Native streaming in the eval harness
The production SSE endpoint simulates streaming (§4), so its timings are not usable. The eval harness calls the Groq API with native streaming and timestamps each received chunk with a monotonic clock. TTFT, TTFS, inter-token latency and tokens/sec all come from these timestamps. The production endpoint is not changed (§3); R8 recommends whether to switch it.

### 9.6 Statistics
- Every configuration answers the same questions, so comparisons are **paired**.
- For each ★ metric, use a paired bootstrap over per-question differences: 10,000 resamples, 95% CI.
- Call a configuration better only when the CI excludes 0. Otherwise report "no detectable difference at n = …".
- S4 compares 3 configurations to the baseline. Apply Holm correction across those 3 and say so in the report.
- Ingestion metrics (S0) are descriptive: report point estimates with bootstrap CIs over pages, and n pages per stratum.
- Non-★ metrics are reported as point estimates, labeled diagnostic.

### 9.7 CI gate
The real corpus and golden sets can't be in the repo (D11), so CI runs on **synthetic documents**.
- **Synthetic fixtures:** the agent creates 4 synthetic loan PDFs with invented lenders, borrowers and figures, in the style of the synthetic document already used by `scripts/ab_eval.py`:
  - one prose agreement
  - one with a fee table and an amortization table
  - one with form fields
  - one rendered to an image-only PDF, to stand in for a scan

  Their Textract output is cached once and committed. A 20-question smoke set over them follows the §5.2 schema and covers every question type. Kousik reviews the documents and questions before they're committed (~1h).
- A GitHub Actions job on every PR runs **retrieval and chunk metrics only** on the synthetic smoke set. No LLM calls, no AWS.
- It fails the build if `recall@budget`, `mrr@10` or entity recall drops more than 0.05 below `evals/baseline.json` for the production configuration, or if the embedder truncation rate rises above its baseline.
- **What the gate does and doesn't show:** it catches code regressions in chunking and retrieval. It says nothing about performance on real documents. Never quote CI numbers as results.
- Updating `baseline.json` is a deliberate change made in its own commit, with the reason in the commit message.
- Ingestion metrics (S0), generation evals (S4) and the load test (S5) run on the real corpus locally only, since they need the private data and AWS or Groq credentials, and cost money.

## 10. Requirements

**Functional**
- F1: Every chunker emits provenance (doc_id, page range, char offsets). A unit test checks offsets against the cleaned text for every chunker.
- F2: All retrievers share one interface: `retrieve(question, k) -> list[(chunk_id, score, rank)]`.
- F3: The QA validator fails if any `span_text` isn't found in the cleaned page text, if any listed `entity` isn't found in the evidence spans or reference answer, if a `qid` is duplicated, or if a split's type mix drifts more than ±1 question from proportional.
- F4: The ingestion-label validator fails if a table's cell grid is inconsistent (overlapping spans, missing cells), if a `reading_order` entry references a missing element, or if a bounding box falls outside the page.
- F5: The runner never calls Textract, and never re-embeds a cached chunk set.
- F6: Any test-split run is recorded with a `test_run: true` tag. The S4 report states how many times the test split was run (target: once).
- F7: `report.py` generates the README results tables and charts from MLflow runs. Numbers are never typed in by hand.
- F8: Every metric in §8 has a unit test on a small hand-built fixture with a known answer (e.g. a 3-chunk ranking with known MRR, a 2×2 table with known cell accuracy, a string pair with known CER).
- F9: A test fails if git tracks any private data: golden-set JSONL, page labels or images, real-corpus extraction output, or per-question results JSONL. It checks `git ls-files` against the private paths and file patterns in D11. `.gitignore` covers the same paths.
- F10: `report.py` output contains aggregate numbers only. It never includes question text, answers, evidence spans or document excerpts from the real corpus. Any worked example in the README comes from the synthetic fixtures.

**Non-functional**
- N1: S0–S3 complete on a laptop CPU, with no GPU required.
- N2: Production behavior is unchanged unless R0 changes it. The default config must reproduce C1 + `R-dense` exactly, checked with a test that compares chunk IDs and text against the pre-refactor output on one fixture document.
- N3: The 122 existing tests keep passing. New deterministic code is tested with pytest. Judge-graded behavior is covered by the calibration in §8.6, not unit tests.
- N4: Secrets come only from environment variables. Real-corpus data stays out of the repo (D11, F9, F10).
- N5: Timing uses a monotonic clock, measured in the client process. Network time is included and stated as such.

## 11. Pre-registered Hypotheses
Write these into the results before S0 runs, and report each as supported, not supported, or inconclusive. Don't add hypotheses after seeing results without labeling them post-hoc.
- **H1:** C4 (layout-aware) beats C1 on `recall@budget` for `table` questions.
- **H2:** `R-hybrid` beats `R-dense` on `lookup` questions that contain exact identifiers (rates, amounts, clause numbers).
- **H3:** Reranking improves `mrr@10` for every base retriever.
- **H4:** `R-pc` improves faithfulness on `explain` questions compared with the same chunker under `R-dense`.
- **H5:** The best configuration's $ per 1k queries is within 20% of the baseline's, because the context budget is fixed.
- **H6:** Scanned pages have at least twice the CER of digital pages.
- **H7:** Test questions from the lowest ingestion-score tercile have lower answer correctness than those from the highest tercile.
- **H8:** C1's embedder truncation rate is above 50%, and C2–C5's are below 5%.

## 12. Milestones

| # | Deliverable | Est. |
|---|---|---|
| R0 | Fix the retriever-cache eviction bug, with a regression test. Measure MiniLM truncation: what fraction of current C1 chunks exceed the model's max sequence length (256 word-piece tokens for all-MiniLM-L6-v2; confirm from the loaded model's `max_seq_length`), and how much text that drops. Record the finding | 2h |
| R1 | Textract re-extraction with LAYOUT (D6) and extraction cache. Ingestion labeling script and validator. Kousik labels 30 pages (5 from scratch). §8.1 metrics with F8 tests. **Run S0** | 8h agent + ~6h Kousik |
| R2 | Chunker interface with provenance; C1 refactored onto it (N2 test); C2–C5; §8.2 chunk metrics; tests | 10h |
| R3 | QA golden-set schema, validator, drafting script, dev/test split. Kousik verifies 120 questions | 4h agent + ~6h Kousik |
| R4 | §8.3 retrieval metrics, runner, chunk/embedding cache, MLflow logging, report generator. **Run S1** | 7h |
| R5 | `R-bm25`, `R-hybrid`, `R-pc`, rerank; index-type study. **Run S2 and S3** | 8h |
| R6 | Generation harness with native streaming and §8.5 telemetry; citation prompt; judge with §8.4 metrics; calibration (Kousik labels 50). **Run S4 on test** | 8h agent + ~2h Kousik |
| R7 | Load test (§9.4). **Run S5.** Paired bootstrap + Holm; synthetic fixtures and smoke set (Kousik reviews); CI smoke gate and `baseline.json` | 9h agent + ~1h Kousik |
| R8 | §8.7 analyses (quality/latency/cost chart, error attribution, confidence vs accuracy); hypothesis results; README results section generated by `report.py`; written rollout recommendation (does the winner fit on t2.micro? should production switch to native streaming?) | 4h |

**Total: ~60h agent + ~15h Kousik.**

The private-data rules (D11) apply from R1 onward: `.gitignore` entries and the F9 test land in R1, before any real labels exist.
**Minimum demonstrable slice:** R0–R4. That gives a measured picture of parsing quality, five chunkers with their own metrics, a span-labeled QA set, and S1 retrieval results with confidence intervals. S1 already answers whether layout-aware chunking helps.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Overfitting choices to the golden set | Pick on dev, report on test once (F6). Pre-registered hypotheses (§11) |
| Too many metrics: some will "win" by chance | One decision metric per stage (D3). CIs only on ★ metrics. Everything else is labeled diagnostic |
| QA set too small for per-type claims (~13–17 test questions per type) | Report per-type n beside every per-type number. Make overall claims only where CIs support them |
| Ingestion set too small for per-stratum claims (6–10 pages each) | Report n pages per stratum. Treat per-stratum results as descriptive |
| Pre-filled ingestion labels biased toward Textract | 5 pages labeled from scratch first (§5.4); report them separately |
| The PDF text layer is itself wrong or out of order | Text-layer sanity checks (§8.1). Numeric fidelity uses multisets, so order doesn't matter. Report excluded pages |
| LLM-drafted questions biased toward easy, extractive lookups | Fixed type quotas (§5.1). Kousik writes some `cross_doc` and `calculation` questions himself |
| Judge disagrees with humans | κ ≥ 0.6 gate (§8.6). Numeric exact match and refusal correctness don't depend on the judge |
| Chunk-size bias in `recall@k` | `recall@budget` is the decision metric (§8.3) |
| MiniLM truncation confounds the chunker comparison | R0 measures it. C2–C5 cap chunks at 256 MiniLM tokens. Report truncation next to each chunker's results (H8) |
| Provider latency varies by time and load | S5 runs configurations back to back in alternating order (§9.4). Report the run time. Latency claims are about this provider at this time, not absolute |
| Rate limits during the load test | Concurrency kept low; 429s counted separately (§9.4) |
| Textract LAYOUT re-extraction cost | Estimate the cost before running (pages × per-page price); run once and cache; cap in D6 |
| Groq model retired mid-project (this has happened before) | Record the exact model ID per run. If retired, re-run S4 and S5 on the replacement and report both |
| Refactoring C1 changes production behavior | N2 equality test against pre-refactor output |
| Cross-encoder too heavy for production | Out of scope here. R8 measures memory and latency and recommends; no deployment |
| Real loan text leaks into the public repo | Private storage and gitignore (D11); F9 test fails the build if private files are tracked; F10 keeps reports aggregate-only |
| Synthetic CI documents don't behave like real ones | The CI gate is only a regression check (§9.7). All reported results come from local runs on the real corpus |

## 14. Decisions

These are fixed for the build. If one turns out to be wrong in practice, stop and raise it. Don't work around it.

**D1. Golden sets.** QA: 120 questions, 40 dev / 80 test, type quotas as in §5.1, split fixed before S1 and never reshuffled. Ingestion: 30 pages with the stratum quotas in §5.4, 5 of them labeled from scratch.

**D2. Evidence labeling.** By page and span text, with ≥ 50% character overlap counting as coverage (§8.3). Never by chunk ID.

**D3. Decision metrics.** Exactly one per stage:
- S1 and S2: `recall@budget`
- S4: answer correctness, with faithfulness as a guard (a configuration whose faithfulness CI is entirely below the baseline's can't win)
- Ties (CI includes 0) go to the configuration with lower $ per 1k queries, then lower p95 end-to-end latency

The ★ metrics are headline numbers reported with CIs. All other metrics are diagnostic.

**D4. Models.**
- Embedder: all-MiniLM-L6-v2 for every configuration.
- Generator: the production Groq model (`llama-3.3-70b-versatile`, or its replacement if retired; record the actual ID).
- Judge: a model from a **different family** than the generator, to avoid a model grading its own style. Default: `openai/gpt-oss-120b` on Groq. Confirm it's available with a live call before R6.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**D5. Citations.** The eval harness prompt asks the model to cite `[chunk_id]` after each claim. This is used for citation precision and recall, and to help the judge. Production prompts don't change in this initiative; R8 recommends whether to adopt it.

**D6. Re-extraction.** Re-run Textract once on all ~60 documents with TABLES + FORMS + LAYOUT, and cache the results. Keep the existing TABLES + FORMS output too, for the S0 comparison. Before running, print the estimated cost; if it's over $15, stop and ask.

**D7. Spend cap.** For LLM calls: $5 per runner invocation, $10 for the S5 load test, $40 total across the initiative. The runner tracks cumulative spend in a local ledger file and refuses to exceed the total without an explicit override flag.

**D8. Where it runs.** All experiments run locally. Nothing is deployed to EC2 in this initiative.

**D9. Build scope.** Build R0–R4, then stop for review before R5.

**D10. Choices left to the agent.** Decide these, then record each in a "Decision log" at the end of this doc with a one-line reason:
- the BM25 library (must be in-process and pip-installable)
- the TEDS implementation and the PDF text-layer library
- text normalization rules for span matching, CER, entity matching and numeric parsing (case, whitespace, punctuation, currency symbols, thousands separators such as Indian lakh grouping)
- the HNSW and IVF parameters for the index study
- the parent window size for `R-pc` when no C4 section is available
- the judge rubric wording (versioned; changes after calibration get a new version)
- the exact MLflow metric names, as long as they map one-to-one onto §8
- how the 20-question CI subset is chosen (it must include every question type)
- the ingestion labeling interface (a script plus a JSON editor is enough; no web app required)

**D11. Private data.** No text, images or labels from the real loan corpus go into the public repo. This covers:
- the QA golden set and the ingestion page labels and images
- real-corpus extraction output and chunk/embedding caches
- per-question and per-page results JSONL
- question text, answers and document excerpts in any report or README

Storage:
- **Source of truth:** a private S3 location with Block Public Access on (a dedicated prefix in the project's existing bucket, or a separate bucket; the agent decides and records it).
- **Working copies:** a local, gitignored directory, synced with a small script (`pull` / `push`). Golden sets are versioned by a version field in the files plus S3 object versioning, not by git.
- MLflow artifacts already go to S3 and may hold per-question results. Confirm the artifact bucket is private before R4, and record that in the decision log.

Committed to the repo: code, schemas, validators, configs, synthetic fixtures (§9.7), and aggregate results.

## 15. Open Questions (for Kousik)
1. ~~Can the loan PDFs' text appear in the public repo?~~ **Resolved 2026-11-03: no.** See D11 and §9.7.
2. ~~Are the cost caps right?~~ **Resolved 2026-11-03: yes.** $15 for re-extraction (D6), $40 for LLM calls including $10 for the load test (D7).
3. ~~Should C2–C5 also run with the 0.25 score threshold?~~ **Resolved 2026-11-03: no.** The threshold applies to `R-dense` only (§7), and unanswerable questions are scored at the generation stage (§8.4).
4. **Does the corpus contain scanned and multi-column pages?** If there are fewer than 6 of either, §5.4's fallback applies, and H6 may be untestable. Consider adding a few public scanned loan documents.

## 16. What This Enables on the CV (rules)
Numbers can be quoted only after the stage that produces them has run on its final data: S0 for parsing, S4 on the test split for quality, S5 for latency percentiles. Each needs its n and, for ★ metrics, its CI. For example: "layout-aware chunking raised recall at a fixed 1,200-word context budget from X to Y (n = 80, 95% CI …)", or "p95 time-to-first-token of X ms over 500 requests". Hypotheses that weren't supported are reported as such; they're still good interview material. No claim may rest on the dev split, a diagnostic metric alone, or the synthetic CI set. Example questions or answers shown in interviews or the README come from the synthetic documents, never the real corpus.

## Decision Log
*(The agent adds entries here for D10 choices: date, choice, one-line reason.)*
