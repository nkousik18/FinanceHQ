# S1 Results — Retrieval (dev split)

**Generated** by `venv/bin/python -m evals.report --stage S1 --split dev` from
`data/results/r4/s1/dev/` (runs: `venv/bin/python -m evals.run --config …`). Do not edit
numbers by hand (F7). Definitions: `evals/retrieval/metrics.py`; statistics: `evals/report.py` docstring;
decisions: `docs/18_session_log.md` (S-2026-11-22-*).

★ = headline metric (95% bootstrap CI over questions). Everything else is diagnostic (point estimate).
Unanswerable questions are left out of coverage metrics. Baseline = `C1-dense` (production configuration).

## Headline

| Run | Chunker | Retriever | top_k | ★ recall@budget (n=33) | ★ entity recall | mrr@10 | hit@10 | context words | ★ evidence fragmentation | retrieval ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1-dense | C1 | R-dense | 20 | **0.515** [0.364, 0.667] | **0.615** [0.485, 0.737] | 0.503 | 0.818 | 985 | **0.000** [0.000, 0.000] | 14.0 / 20.6 |
| C2-dense | C2 | R-dense | 20 | **0.773** [0.636, 0.894] | **0.802** [0.695, 0.894] | 0.596 | 0.818 | 1055 | **0.000** [0.000, 0.000] | 9.7 / 16.5 |
| C3-dense | C3 | R-dense | 20 | **0.652** [0.500, 0.788] | **0.681** [0.560, 0.793] | 0.551 | 0.758 | 1034 | **0.000** [0.000, 0.000] | 7.8 / 8.7 |
| C4-dense | C4 | R-dense | 20 | **0.758** [0.606, 0.894] | **0.797** [0.701, 0.883] | 0.589 | 0.788 | 1049 | **0.000** [0.000, 0.000] | 9.3 / 11.1 |
| C5-dense | C5 | R-dense | 20 | **0.758** [0.606, 0.894] | **0.802** [0.696, 0.898] | 0.592 | 0.788 | 1050 | **0.000** [0.000, 0.000] | 11.5 / 27.8 |

## Pick (decision metric: recall@budget)

1. `C2-dense` — recall@budget 0.773 ← **picked**
2. `C4-dense` — recall@budget 0.758 ← **picked**
3. `C5-dense` — recall@budget 0.758
4. `C3-dense` — recall@budget 0.652
5. `C1-dense` — recall@budget 0.515

**Tie at the pick boundary:** `C4-dense`, `C5-dense` have the same recall@budget. Tie rule: faster wins (retrieval latency p50: `C4-dense` 9.3 ms, `C5-dense` 11.5 ms), as in S0.

The pick follows the point estimate (D3). Whether neighbours in the ranking are distinguishable is in the
next table: if the CI includes 0, the pick between them is not supported by the data at this n.

## Paired comparisons

**recall@budget** (paired, per-question differences, 95% bootstrap CI)

| Comparison | Mean difference [95% CI] | p | Holm p (vs baseline) | Verdict |
|---|---|---|---|---|
| `C2-dense` − `C1-dense` | +0.258 [+0.121, +0.409] | < 0.0001 | < 0.0001 | C2-dense better |
| `C3-dense` − `C1-dense` | +0.136 [+0.000, +0.288] | 0.0698 | 0.0698 | no detectable difference (n = 33) |
| `C4-dense` − `C1-dense` | +0.242 [+0.076, +0.409] | 0.0046 | 0.0120 | C4-dense better |
| `C5-dense` − `C1-dense` | +0.242 [+0.076, +0.409] | 0.0040 | 0.0120 | C5-dense better |
| `C2-dense` − `C4-dense` (ranks 1–2) | +0.015 [-0.121, +0.152] | 0.9316 |  | no detectable difference (n = 33) |
| `C4-dense` − `C5-dense` (ranks 2–3) | +0.000 [-0.152, +0.152] | 1.0000 |  | no detectable difference (n = 33) |
| `C5-dense` − `C3-dense` (ranks 3–4) | +0.106 [-0.030, +0.242] | 0.1484 |  | no detectable difference (n = 33) |
| `C3-dense` − `C1-dense` (ranks 4–5) | +0.136 [+0.000, +0.288] | 0.0698 |  | no detectable difference (n = 33) |

**entity_recall** (paired, per-question differences, 95% bootstrap CI)

| Comparison | Mean difference [95% CI] | p | Holm p (vs baseline) | Verdict |
|---|---|---|---|---|
| `C2-dense` − `C1-dense` | +0.187 [+0.086, +0.303] | < 0.0001 | < 0.0001 | C2-dense better |
| `C3-dense` − `C1-dense` | +0.066 [-0.076, +0.202] | 0.3658 | 0.3658 | no detectable difference (n = 33) |
| `C4-dense` − `C1-dense` | +0.182 [+0.045, +0.318] | 0.0124 | 0.0248 | C4-dense better |
| `C5-dense` − `C1-dense` | +0.187 [+0.061, +0.318] | 0.0028 | 0.0084 | C5-dense better |
| `C2-dense` − `C4-dense` (ranks 1–2) | +0.005 [-0.091, +0.101] | 0.8948 |  | no detectable difference (n = 33) |
| `C4-dense` − `C5-dense` (ranks 2–3) | -0.005 [-0.096, +0.081] | 0.9128 |  | no detectable difference (n = 33) |
| `C5-dense` − `C3-dense` (ranks 3–4) | +0.121 [+0.005, +0.247] | 0.0346 |  | C5-dense better |
| `C3-dense` − `C1-dense` (ranks 4–5) | +0.066 [-0.076, +0.202] | 0.3658 |  | no detectable difference (n = 33) |

## recall@budget by question type

| Group | n | `C1-dense` | `C2-dense` | `C3-dense` | `C4-dense` | `C5-dense` |
|---|---|---|---|---|---|---|
| type = lookup | 8 | 0.375 | 0.625 | 0.500 | 0.750 | 0.750 |
| type = table | 8 | 0.625 | 0.875 | 0.875 | 0.875 | 1.000 |
| type = calculation | 7 | 0.786 | 0.857 | 0.857 | 0.857 | 0.714 |
| type = cross_doc | 5 | 0.300 | 0.700 | 0.600 | 0.600 | 0.600 |
| type = explain | 5 | 0.400 | 0.800 | 0.300 | 0.600 | 0.600 |

## recall@budget by corpus part

| Group | n | `C1-dense` | `C2-dense` | `C3-dense` | `C4-dense` | `C5-dense` |
|---|---|---|---|---|---|---|
| part = edgar | 21 | 0.357 | 0.690 | 0.500 | 0.643 | 0.690 |
| part = consumer | 9 | 0.833 | 0.889 | 0.889 | 0.944 | 0.833 |
| part = simulated_scan | 3 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 |

## Entity recall by question type

| Group | n | `C1-dense` | `C2-dense` | `C3-dense` | `C4-dense` | `C5-dense` |
|---|---|---|---|---|---|---|
| type = lookup | 8 | 0.646 | 0.896 | 0.708 | 0.896 | 0.896 |
| type = table | 8 | 0.750 | 0.938 | 0.812 | 0.938 | 0.938 |
| type = calculation | 7 | 0.590 | 0.590 | 0.590 | 0.590 | 0.495 |
| type = cross_doc | 5 | 0.367 | 0.733 | 0.700 | 0.767 | 0.733 |
| type = explain | 5 | 0.633 | 0.800 | 0.533 | 0.733 | 0.933 |

## Diagnostic metrics (overall)

| Metric (diagnostic) | `C1-dense` | `C2-dense` | `C3-dense` | `C4-dense` | `C5-dense` |
|---|---|---|---|---|---|
| recall@1 | 0.303 | 0.379 | 0.364 | 0.409 | 0.409 |
| recall@3 | 0.530 | 0.667 | 0.591 | 0.682 | 0.591 |
| recall@5 | 0.621 | 0.727 | 0.636 | 0.697 | 0.667 |
| recall@10 | 0.758 | 0.773 | 0.697 | 0.742 | 0.758 |
| hit@1 | 0.364 | 0.455 | 0.424 | 0.455 | 0.485 |
| hit@5 | 0.697 | 0.788 | 0.697 | 0.758 | 0.697 |
| hit@10 | 0.818 | 0.818 | 0.758 | 0.788 | 0.788 |
| precision@1 | 0.364 | 0.455 | 0.424 | 0.455 | 0.485 |
| precision@5 | 0.200 | 0.188 | 0.164 | 0.170 | 0.170 |
| precision@10 | 0.127 | 0.106 | 0.097 | 0.091 | 0.097 |
| context_precision | 0.459 | 0.579 | 0.546 | 0.586 | 0.566 |
| mrr@10 | 0.503 | 0.596 | 0.551 | 0.589 | 0.592 |
| ndcg@10 | 0.465 | 0.609 | 0.558 | 0.609 | 0.612 |
| map@10 | 0.362 | 0.538 | 0.495 | 0.548 | 0.549 |
| noise_ratio | 0.724 | 0.849 | 0.798 | 0.846 | 0.868 |
| redundancy | 0.493 | 0.332 | 0.343 | 0.310 | 0.314 |
| context_words | 985 | 1055 | 1034 | 1049 | 1050 |
| cross_doc_coverage | 0.000 | 0.400 | 0.200 | 0.400 | 0.400 |
| amend_coverage | 0.000 | 0.667 | 0.333 | 0.333 | 0.667 |
| amend_only_superseded | 0.667 | 0.333 | 0.667 | 0.333 | 0.333 |

## Notes

- n = 33 answerable dev questions (7 unanswerable left out of coverage metrics). Type and part breakdowns have single-digit n: read them as diagnostics, not results.
- Cross-document coverage (n = 5) and amendment coverage (n = 3) are too small to compare chunkers; H9 is judged on the full set in R8.
- The test split is untouched: S1–S3 pick on dev only (§9.3).
- **Evidence fragmentation is 0 for every chunker.** QA evidence spans are short (about one sentence), so each span lies at least 50% inside a single chunk of every chunker. On this question set the metric does not separate chunkers.

## Provenance

Questions file sha256 `ddca7f3149032593…`, manifest `5daa8444816306f5…`, embedder `sentence-transformers/all-MiniLM-L6-v2|normalized|float32`, parse `P2` (scans `P2V`), cleaner `v2`.

| Run | git SHA | questions | cache | MLflow run | seconds |
|---|---|---|---|---|---|
| `C1-dense` | 938ea7b4005b | 40 | {'built': 44, 'loaded': 11} | 50051ca2bcf4 | 84.5 |
| `C2-dense` | 938ea7b4005b | 40 | {'built': 55, 'loaded': 0} | 91af783ac210 | 165.4 |
| `C3-dense` | 938ea7b4005b | 40 | {'built': 0, 'loaded': 55} | cd277f3ed9cf | 1.0 |
| `C4-dense` | 938ea7b4005b | 40 | {'built': 55, 'loaded': 0} | 506b3f4ce55e | 176.2 |
| `C5-dense` | 938ea7b4005b | 40 | {'built': 55, 'loaded': 0} | da48a11c97c2 | 391.9 |
