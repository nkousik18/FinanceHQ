# R2 Results — Chunk Metrics (§8.2)

**Generated** by `venv/bin/python -m evals.chunking.r2_report` from `data/results/r2/chunk_metrics.json`
(run: `venv/bin/python -m evals.chunking.run_metrics`). Do not edit numbers by hand (F7).

Corpus: 92 documents (41 EDGAR, 11 consumer, 40 simulated scans); parse `downstream_config("P2", part)`,
cleaner `v2`. Definitions: `evals/chunking/metrics.py` docstring and the scope Decision Log
(2026-11-13). Decisions and caveats: `docs/18_session_log.md` (S-2026-11-13-4 … 9).
★ = headline metric, with document-bootstrap 95% CIs (documents as units).

## 1. All documents
| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Chunks | 4,774 | 9,760 | 6,583 | 8,749 | 10,744 |
| MiniLM tokens p5/p50/p95/max | 380 / 476 / 711 / 1,954 | 77 / 176 / 200 / 200 | 253 / 254 / 254 / 254 | 70 / 208 / 254 / 254 | 54 / 139 / 200 / 200 |
| tiktoken p5/p50/p95/max | 381 / 483 / 699 / 1,020 | 75 / 175 / 210 / 288 | 238 / 253 / 279 / 378 | 74 / 210 / 261 / 360 | 53 / 141 / 203 / 288 |
| Words p5/p50/p95/max | 301 / 358 / 531 / 600 | 55 / 129 / 162 / 190 | 159 / 194 / 214 / 242 | 48 / 154 / 200 / 231 | 38 / 104 / 160 / 186 |
| ★ Truncation rate | **98.99%** [98.06, 99.39] | 0% | 0% | 0% | 0% |
| Tokens dropped per chunk copy | 49.87% | 0% | 0% | 0% | 0% |
| ★ Page words never embedded | **22.96%** [22.48, 23.39] | 0% | 0% | **0.26%** [0.2, 0.38] | 0% |
| Unaligned words (C1 alignment gap) | 2,178 | 0 | 0 | 0 | 0 |
| Tables (header + data row spans) | 216 | 216 | 216 | 216 | 216 |
| Tables split | 12.96% | 49.07% | 46.76% | 34.72% | 56.94% |
| Table-break rate (split, header not repeated) | 12.96% | 49.07% | 46.76% | 0% | 56.94% |
| Sentence-break rate | 6.94% | 12.32% | 97.57% | 8.02% | 10.42% |
| Overlap redundancy | 38.84% | 8.39% | 12.55% | 5.03% | 0% |
| ★ Evidence fragmentation | pending R3 | pending R3 | pending R3 | pending R3 | pending R3 |

## 2. Index size and build time
One FAISS IndexFlatIP per chunker over the corpus; production MiniLM, normalized, batch 64, CPU.

| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Vectors | 4,774 | 9,760 | 6,583 | 8,749 | 10,744 |
| Index memory (MB) | 6.99 | 14.3 | 9.64 | 12.82 | 15.74 |
| Index disk (MB) | 6.99 | 14.3 | 9.64 | 12.82 | 15.74 |
| Chunking time, total (s) | 0.5 | 9.6 | 4.8 | 8.2 | 163.4 |
| Embedding time, total (s) | 82.9 | 126.5 | 101.8 | 122.5 | 115.1 |
| Embedding time per doc (s) | 0.9 | 1.38 | 1.11 | 1.33 | 1.25 |

## 3. By corpus part
### consumer (11 docs)
| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Chunks | 77 | 108 | 79 | 186 | 125 |
| ★ Truncation rate | **92.21%** [83.67, 96.94] | 0% | 0% | 0% | 0% |
| ★ Page words never embedded | **24%** [19.07, 27.52] | 0% | 0% | **3.44%** [2.79, 4.2] | 0% |
| Table-break rate | 2.74% (n=73) | 35.62% (n=73) | 23.29% (n=73) | 0% (n=73) | 38.36% (n=73) |
| Sentence-break rate | 0% | 0% | 88.97% | 0% | 0% |
| Overlap redundancy | 38.29% | 15.34% | 11.57% | 1.41% | 0% |

### edgar (41 docs)
| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Chunks | 4,595 | 9,481 | 6,382 | 8,386 | 10,432 |
| ★ Truncation rate | **99.56%** [99.16, 99.75] | 0% | 0% | 0% | 0% |
| ★ Page words never embedded | **22.99%** [22.52, 23.43] | 0% | 0% | **0.22%** [0.16, 0.31] | 0% |
| Table-break rate | 24.3% (n=107) | 63.55% (n=107) | 67.29% (n=107) | 0% (n=107) | 72.9% (n=107) |
| Sentence-break rate | 7.04% | 12.46% | 97.72% | 8.24% | 10.55% |
| Overlap redundancy | 38.96% | 8.29% | 12.61% | 5.11% | 0% |

### simulated_scan (40 docs)
| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Chunks | 102 | 171 | 122 | 177 | 187 |
| ★ Truncation rate | **78.43%** [72.28, 84.41] | 0% | 0% | 0% | 0% |
| ★ Page words never embedded | **20.85%** [18.01, 23.78] | 0% | 0% | **1.05%** [0.58, 1.64] | 0% |
| Table-break rate | 0% (n=36) | 33.33% (n=36) | 33.33% (n=36) | 0% (n=36) | 47.22% (n=36) |
| Sentence-break rate | 7.02% | 11.45% | 93.29% | 5.28% | 9.52% |
| Overlap redundancy | 31.88% | 9.43% | 9.76% | 2.65% | 0% |


## Notes and caveats

- **H8** (C1 truncation > 50%, C2–C5 < 5%) is **supported** by these measurements; the formal
  hypothesis write-up is in R8.
- Sentence-break uses C2's splitter (syntok per line) as the reference, so C2 and C5 only break
  sentences they split for size: they are favoured by construction.
- C1 duplicates tables (page text + markdown); its redundancy includes that. Its low table-break
  rate comes from the markdown copy being kept whole.
- C4's never-embedded words are page headers, footers and page numbers, left out on purpose.
- C1 unaligned words (in truncated chunks, not found in the chunk text) count as not embedded;
  counting them as embedded instead changes C1's rate by at most unaligned ÷ page words.
- C4 = Docling headings **plus the run-in heading rule** (S-2026-11-13-10, a recorded deviation
  from §6.1); strict C4 is `C4LayoutAware(runin_headings=False)`.
- C2 overlap is whole sentences only (46% of boundaries have none); C5 is the literal p95 rule
  (89% of its cuts come from the size limit). Both kept as built by decision (S-2026-11-13-11, 12).
- Evidence fragmentation (★) needs the QA set: pending R3.
- Timings are one laptop CPU run (Apple Silicon, 8 cores): rough, not a benchmark.

## 4. Chunker parameters

- **C1:** `{"chunk_size": 400, "overlap": 60, "words_per_token": 0.75}`
- **C2:** `{"max_tokens": 200, "overlap": 40, "splitter": "syntok 1.4.4", "tokenizer": "all-MiniLM-L6-v2"}`
- **C3:** `{"size": 256, "overlap": 32, "special_tokens": 2, "content_tokens": 254, "tokenizer": "all-MiniLM-L6-v2"}`
- **C4:** `{"max_tokens": 254, "min_tokens": 50, "overlap": 40, "prefix_max": 64, "runin_headings": true, "tokenizer": "all-MiniLM-L6-v2"}`
- **C5:** `{"percentile": 95.0, "max_tokens": 200, "min_tokens": 50, "overlap": 0, "sentence_embedder": "all-MiniLM-L6-v2 (normalized, per unit)", "splitter": "syntok 1.4.4", "tokenizer": "all-MiniLM-L6-v2"}`
