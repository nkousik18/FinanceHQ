# S2 Results — Retrieval (dev split)

**Generated** by `venv/bin/python -m evals.report --stage S2 --split dev` from
`data/results/r4/s2/dev/` (runs: `venv/bin/python -m evals.run --config …`). Do not edit
numbers by hand (F7). Definitions: `evals/retrieval/metrics.py`; statistics: `evals/report.py` docstring;
decisions: `docs/18_session_log.md` (S-2026-11-22-*).

★ = headline metric (95% bootstrap CI over questions). Everything else is diagnostic (point estimate).
Unanswerable questions are left out of coverage metrics. Baseline = `C1-dense` (production configuration).

## Headline

| Run | Chunker | Retriever | top_k | ★ recall@budget (n=33) | ★ entity recall | mrr@10 | hit@10 | context words | ★ evidence fragmentation | retrieval ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1-dense (S1) | C1 | R-dense | 20 | **0.515** [0.364, 0.667] | **0.615** [0.485, 0.737] | 0.503 | 0.818 | 985 | **0.000** [0.000, 0.000] | 14.0 / 20.6 |
| C2-dense | C2 | R-dense | 20 | **0.773** [0.636, 0.894] | **0.802** [0.695, 0.894] | 0.596 | 0.818 | 1055 | **0.000** [0.000, 0.000] | 10.5 / 15.4 |
| C2-dense+rerank | C2 | R-dense+rerank | 20 | **0.833** [0.712, 0.939] | **0.832** [0.753, 0.904] | 0.724 | 0.879 | 1072 | **0.000** [0.000, 0.000] | 391.5 / 1038.9 |
| C2-bm25 | C2 | R-bm25 | 20 | **0.833** [0.712, 0.939] | **0.792** [0.687, 0.887] | 0.710 | 0.879 | 1047 | **0.000** [0.000, 0.000] | 0.1 / 0.2 |
| C2-bm25+rerank | C2 | R-bm25+rerank | 20 | **0.833** [0.712, 0.939] | **0.817** [0.722, 0.903] | 0.680 | 0.909 | 1067 | **0.000** [0.000, 0.000] | 343.6 / 514.3 |
| C2-hybrid | C2 | R-hybrid | 20 | **0.818** [0.697, 0.939] | **0.817** [0.717, 0.903] | 0.737 | 0.879 | 1056 | **0.000** [0.000, 0.000] | 12.2 / 14.5 |
| C2-hybrid+rerank | C2 | R-hybrid+rerank | 20 | **0.864** [0.742, 0.955] | **0.847** [0.771, 0.919] | 0.712 | 0.939 | 1051 | **0.000** [0.000, 0.000] | 327.7 / 770.3 |
| C4-dense | C4 | R-dense | 20 | **0.758** [0.606, 0.894] | **0.797** [0.701, 0.883] | 0.589 | 0.788 | 1049 | **0.000** [0.000, 0.000] | 9.5 / 11.5 |
| C4-dense+rerank | C4 | R-dense+rerank | 20 | **0.773** [0.621, 0.909] | **0.782** [0.676, 0.878] | 0.641 | 0.788 | 1049 | **0.000** [0.000, 0.000] | 452.7 / 983.2 |
| C4-bm25 | C4 | R-bm25 | 20 | **0.803** [0.667, 0.924] | **0.767** [0.646, 0.874] | 0.699 | 0.818 | 1044 | **0.000** [0.000, 0.000] | 0.1 / 0.2 |
| C4-bm25+rerank | C4 | R-bm25+rerank | 20 | **0.803** [0.667, 0.924] | **0.802** [0.696, 0.898] | 0.654 | 0.879 | 1048 | **0.000** [0.000, 0.000] | 392.6 / 744.8 |
| C4-hybrid | C4 | R-hybrid | 20 | **0.818** [0.682, 0.939] | **0.822** [0.726, 0.907] | 0.655 | 0.848 | 1036 | **0.000** [0.000, 0.000] | 10.0 / 16.6 |
| C4-hybrid+rerank | C4 | R-hybrid+rerank | 20 | **0.803** [0.667, 0.924] | **0.812** [0.712, 0.902] | 0.676 | 0.848 | 1043 | **0.000** [0.000, 0.000] | 605.4 / 1214.5 |
| C2-pc | C2 | R-pc (parents C4) | 20 | **0.727** [0.576, 0.864] | **0.721** [0.586, 0.842] | 0.632 | 0.788 | 908 | **0.000** [0.000, 0.000] | 16.5 / 31.6 |
| C2-pc+rerank | C2 | R-pc+rerank (parents C4) | 20 | **0.667** [0.515, 0.818] | **0.686** [0.561, 0.803] | 0.685 | 0.879 | 841 | **0.000** [0.000, 0.000] | 916.8 / 1415.8 |

## Pick (decision metric: recall@budget)

1. `C2-hybrid+rerank` — recall@budget 0.864 ← **picked**
2. `C2-bm25` — recall@budget 0.833 ← **picked**
3. `C2-bm25+rerank` — recall@budget 0.833 ← **picked**
4. `C2-dense+rerank` — recall@budget 0.833
5. `C4-hybrid` — recall@budget 0.818
6. `C2-hybrid` — recall@budget 0.818
7. `C4-bm25` — recall@budget 0.803
8. `C4-bm25+rerank` — recall@budget 0.803
9. `C4-hybrid+rerank` — recall@budget 0.803
10. `C2-dense` — recall@budget 0.773
11. `C4-dense+rerank` — recall@budget 0.773
12. `C4-dense` — recall@budget 0.758
13. `C2-pc` — recall@budget 0.727
14. `C2-pc+rerank` — recall@budget 0.667

**Tie at the pick boundary:** `C2-bm25`, `C2-bm25+rerank`, `C2-dense+rerank` have the same recall@budget. Tie rule: faster wins (retrieval latency p50: `C2-bm25` 0.1 ms, `C2-bm25+rerank` 343.6 ms, `C2-dense+rerank` 391.5 ms), as in S0.

The pick follows the point estimate (D3). Whether neighbours in the ranking are distinguishable is in the
next table: if the CI includes 0, the pick between them is not supported by the data at this n.

## Paired comparisons

**recall@budget** (paired, per-question differences, 95% bootstrap CI)

| Comparison | Mean difference [95% CI] | p | Holm p (vs baseline) | Verdict |
|---|---|---|---|---|
| `C2-dense` − `C1-dense` | +0.258 [+0.121, +0.409] | < 0.0001 | < 0.0001 | C2-dense better |
| `C2-dense+rerank` − `C1-dense` | +0.318 [+0.152, +0.485] | 0.0002 | 0.0022 | C2-dense+rerank better |
| `C2-bm25` − `C1-dense` | +0.318 [+0.152, +0.485] | 0.0002 | 0.0022 | C2-bm25 better |
| `C2-bm25+rerank` − `C1-dense` | +0.318 [+0.136, +0.500] | 0.0012 | 0.0084 | C2-bm25+rerank better |
| `C2-hybrid` − `C1-dense` | +0.303 [+0.136, +0.470] | 0.0006 | 0.0048 | C2-hybrid better |
| `C2-hybrid+rerank` − `C1-dense` | +0.348 [+0.182, +0.515] | 0.0002 | 0.0022 | C2-hybrid+rerank better |
| `C4-dense` − `C1-dense` | +0.242 [+0.076, +0.409] | 0.0046 | 0.0138 | C4-dense better |
| `C4-dense+rerank` − `C1-dense` | +0.258 [+0.091, +0.424] | 0.0026 | 0.0120 | C4-dense+rerank better |
| `C4-bm25` − `C1-dense` | +0.288 [+0.106, +0.470] | 0.0014 | 0.0084 | C4-bm25 better |
| `C4-bm25+rerank` − `C1-dense` | +0.288 [+0.106, +0.470] | 0.0024 | 0.0120 | C4-bm25+rerank better |
| `C4-hybrid` − `C1-dense` | +0.303 [+0.152, +0.455] | < 0.0001 | < 0.0001 | C4-hybrid better |
| `C4-hybrid+rerank` − `C1-dense` | +0.288 [+0.121, +0.455] | < 0.0001 | < 0.0001 | C4-hybrid+rerank better |
| `C2-pc` − `C1-dense` | +0.212 [+0.061, +0.379] | 0.0112 | 0.0224 | C2-pc better |
| `C2-pc+rerank` − `C1-dense` | +0.152 [-0.030, +0.333] | 0.1078 | 0.1078 | no detectable difference (n = 33) |
| `C2-hybrid+rerank` − `C2-bm25` (ranks 1–2) | +0.030 [-0.076, +0.152] | 0.6934 |  | no detectable difference (n = 33) |
| `C2-bm25` − `C2-bm25+rerank` (ranks 2–3) | +0.000 [-0.091, +0.091] | 1.0000 |  | no detectable difference (n = 33) |
| `C2-bm25+rerank` − `C2-dense+rerank` (ranks 3–4) | +0.000 [-0.091, +0.091] | 1.0000 |  | no detectable difference (n = 33) |
| `C2-dense+rerank` − `C4-hybrid` (ranks 4–5) | +0.015 [-0.076, +0.106] | 0.8766 |  | no detectable difference (n = 33) |
| `C4-hybrid` − `C2-hybrid` (ranks 5–6) | +0.000 [-0.091, +0.091] | 1.0000 |  | no detectable difference (n = 33) |
| `C2-hybrid` − `C4-bm25` (ranks 6–7) | +0.015 [-0.091, +0.121] | 0.9128 |  | no detectable difference (n = 33) |
| `C4-bm25` − `C4-bm25+rerank` (ranks 7–8) | +0.000 [-0.076, +0.061] | 1.0000 |  | no detectable difference (n = 33) |
| `C4-bm25+rerank` − `C4-hybrid+rerank` (ranks 8–9) | +0.000 [-0.091, +0.091] | 1.0000 |  | no detectable difference (n = 33) |
| `C4-hybrid+rerank` − `C2-dense` (ranks 9–10) | +0.030 [-0.106, +0.167] | 0.7576 |  | no detectable difference (n = 33) |
| `C2-dense` − `C4-dense+rerank` (ranks 10–11) | +0.000 [-0.136, +0.136] | 1.0000 |  | no detectable difference (n = 33) |
| `C4-dense+rerank` − `C4-dense` (ranks 11–12) | +0.015 [+0.000, +0.045] | 0.7406 |  | no detectable difference (n = 33) |
| `C4-dense` − `C2-pc` (ranks 12–13) | +0.030 [-0.152, +0.212] | 0.8510 |  | no detectable difference (n = 33) |
| `C2-pc` − `C2-pc+rerank` (ranks 13–14) | +0.061 [-0.106, +0.227] | 0.5414 |  | no detectable difference (n = 33) |

**entity_recall** (paired, per-question differences, 95% bootstrap CI)

| Comparison | Mean difference [95% CI] | p | Holm p (vs baseline) | Verdict |
|---|---|---|---|---|
| `C2-dense` − `C1-dense` | +0.187 [+0.086, +0.303] | < 0.0001 | < 0.0001 | C2-dense better |
| `C2-dense+rerank` − `C1-dense` | +0.217 [+0.101, +0.338] | 0.0004 | 0.0048 | C2-dense+rerank better |
| `C2-bm25` − `C1-dense` | +0.177 [+0.020, +0.333] | 0.0324 | 0.1070 | C2-bm25 better |
| `C2-bm25+rerank` − `C1-dense` | +0.202 [+0.056, +0.343] | 0.0078 | 0.0624 | C2-bm25+rerank better |
| `C2-hybrid` − `C1-dense` | +0.202 [+0.071, +0.333] | 0.0032 | 0.0288 | C2-hybrid better |
| `C2-hybrid+rerank` − `C1-dense` | +0.232 [+0.111, +0.359] | 0.0004 | 0.0048 | C2-hybrid+rerank better |
| `C4-dense` − `C1-dense` | +0.182 [+0.045, +0.318] | 0.0124 | 0.0868 | C4-dense better |
| `C4-dense+rerank` − `C1-dense` | +0.167 [+0.035, +0.308] | 0.0226 | 0.1070 | C4-dense+rerank better |
| `C4-bm25` − `C1-dense` | +0.152 [-0.005, +0.308] | 0.0600 | 0.1200 | no detectable difference (n = 33) |
| `C4-bm25+rerank` − `C1-dense` | +0.187 [+0.040, +0.333] | 0.0128 | 0.0868 | C4-bm25+rerank better |
| `C4-hybrid` − `C1-dense` | +0.207 [+0.096, +0.328] | < 0.0001 | < 0.0001 | C4-hybrid better |
| `C4-hybrid+rerank` − `C1-dense` | +0.197 [+0.076, +0.328] | 0.0012 | 0.0120 | C4-hybrid+rerank better |
| `C2-pc` − `C1-dense` | +0.106 [+0.020, +0.202] | 0.0214 | 0.1070 | C2-pc better |
| `C2-pc+rerank` − `C1-dense` | +0.071 [-0.076, +0.222] | 0.3494 | 0.3494 | no detectable difference (n = 33) |
| `C2-hybrid+rerank` − `C2-bm25` (ranks 1–2) | +0.056 [-0.035, +0.162] | 0.2600 |  | no detectable difference (n = 33) |
| `C2-bm25` − `C2-bm25+rerank` (ranks 2–3) | -0.025 [-0.111, +0.045] | 0.5610 |  | no detectable difference (n = 33) |
| `C2-bm25+rerank` − `C2-dense+rerank` (ranks 3–4) | -0.015 [-0.091, +0.045] | 0.9290 |  | no detectable difference (n = 33) |
| `C2-dense+rerank` − `C4-hybrid` (ranks 4–5) | +0.010 [-0.056, +0.081] | 0.7720 |  | no detectable difference (n = 33) |
| `C4-hybrid` − `C2-hybrid` (ranks 5–6) | +0.005 [-0.071, +0.091] | 0.9898 |  | no detectable difference (n = 33) |
| `C2-hybrid` − `C4-bm25` (ranks 6–7) | +0.051 [-0.015, +0.141] | 0.2610 |  | no detectable difference (n = 33) |
| `C4-bm25` − `C4-bm25+rerank` (ranks 7–8) | -0.035 [-0.121, +0.030] | 0.4220 |  | no detectable difference (n = 33) |
| `C4-bm25+rerank` − `C4-hybrid+rerank` (ranks 8–9) | -0.010 [-0.091, +0.061] | 0.9386 |  | no detectable difference (n = 33) |
| `C4-hybrid+rerank` − `C2-dense` (ranks 9–10) | +0.010 [-0.076, +0.101] | 0.8674 |  | no detectable difference (n = 33) |
| `C2-dense` − `C4-dense+rerank` (ranks 10–11) | +0.020 [-0.081, +0.121] | 0.6742 |  | no detectable difference (n = 33) |
| `C4-dense+rerank` − `C4-dense` (ranks 11–12) | -0.015 [-0.045, +0.000] | 0.7230 |  | no detectable difference (n = 33) |
| `C4-dense` − `C2-pc` (ranks 12–13) | +0.076 [-0.051, +0.207] | 0.2662 |  | no detectable difference (n = 33) |
| `C2-pc` − `C2-pc+rerank` (ranks 13–14) | +0.035 [-0.096, +0.172] | 0.5966 |  | no detectable difference (n = 33) |

## recall@budget by question type

| Group | n | `C1-dense` | `C2-dense` | `C2-dense+rerank` | `C2-bm25` | `C2-bm25+rerank` | `C2-hybrid` | `C2-hybrid+rerank` | `C4-dense` | `C4-dense+rerank` | `C4-bm25` | `C4-bm25+rerank` | `C4-hybrid` | `C4-hybrid+rerank` | `C2-pc` | `C2-pc+rerank` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| type = lookup | 8 | 0.375 | 0.625 | 0.750 | 0.750 | 0.750 | 0.625 | 0.875 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.875 | 0.625 | 0.750 |
| type = table | 8 | 0.625 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.750 | 0.500 |
| type = calculation | 7 | 0.786 | 0.857 | 0.929 | 0.929 | 0.929 | 0.857 | 0.929 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.786 |
| type = cross_doc | 5 | 0.300 | 0.700 | 0.700 | 0.800 | 0.700 | 0.800 | 0.700 | 0.600 | 0.700 | 0.700 | 0.600 | 0.800 | 0.600 | 0.600 | 0.600 |
| type = explain | 5 | 0.400 | 0.800 | 0.900 | 0.800 | 0.900 | 1.000 | 0.900 | 0.600 | 0.600 | 0.800 | 0.900 | 0.800 | 0.700 | 0.800 | 0.700 |

## recall@budget by corpus part

| Group | n | `C1-dense` | `C2-dense` | `C2-dense+rerank` | `C2-bm25` | `C2-bm25+rerank` | `C2-hybrid` | `C2-hybrid+rerank` | `C4-dense` | `C4-dense+rerank` | `C4-bm25` | `C4-bm25+rerank` | `C4-hybrid` | `C4-hybrid+rerank` | `C2-pc` | `C2-pc+rerank` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| part = edgar | 21 | 0.357 | 0.690 | 0.786 | 0.738 | 0.786 | 0.738 | 0.833 | 0.643 | 0.643 | 0.690 | 0.714 | 0.714 | 0.714 | 0.595 | 0.524 |
| part = consumer | 9 | 0.833 | 0.889 | 0.889 | 1.000 | 0.889 | 0.944 | 0.889 | 0.944 | 1.000 | 1.000 | 0.944 | 1.000 | 0.944 | 0.944 | 1.000 |
| part = simulated_scan | 3 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |

## Entity recall by question type

| Group | n | `C1-dense` | `C2-dense` | `C2-dense+rerank` | `C2-bm25` | `C2-bm25+rerank` | `C2-hybrid` | `C2-hybrid+rerank` | `C4-dense` | `C4-dense+rerank` | `C4-bm25` | `C4-bm25+rerank` | `C4-hybrid` | `C4-hybrid+rerank` | `C2-pc` | `C2-pc+rerank` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| type = lookup | 8 | 0.646 | 0.896 | 0.938 | 0.812 | 0.812 | 0.812 | 0.938 | 0.896 | 0.833 | 0.750 | 0.750 | 0.875 | 0.875 | 0.833 | 0.896 |
| type = table | 8 | 0.750 | 0.938 | 0.938 | 0.812 | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 | 0.812 | 0.938 | 0.938 | 0.938 | 0.750 | 0.500 |
| type = calculation | 7 | 0.590 | 0.590 | 0.638 | 0.638 | 0.638 | 0.590 | 0.638 | 0.590 | 0.590 | 0.590 | 0.590 | 0.590 | 0.590 | 0.590 | 0.543 |
| type = cross_doc | 5 | 0.367 | 0.733 | 0.800 | 0.867 | 0.800 | 0.867 | 0.800 | 0.767 | 0.767 | 0.867 | 0.867 | 0.767 | 0.867 | 0.600 | 0.767 |
| type = explain | 5 | 0.633 | 0.800 | 0.800 | 0.867 | 0.900 | 0.900 | 0.900 | 0.733 | 0.733 | 0.867 | 0.900 | 0.933 | 0.767 | 0.800 | 0.767 |

## Diagnostic metrics (overall)

| Metric (diagnostic) | `C1-dense` | `C2-dense` | `C2-dense+rerank` | `C2-bm25` | `C2-bm25+rerank` | `C2-hybrid` | `C2-hybrid+rerank` | `C4-dense` | `C4-dense+rerank` | `C4-bm25` | `C4-bm25+rerank` | `C4-hybrid` | `C4-hybrid+rerank` | `C2-pc` | `C2-pc+rerank` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@1 | 0.303 | 0.379 | 0.530 | 0.515 | 0.470 | 0.545 | 0.500 | 0.409 | 0.485 | 0.530 | 0.470 | 0.485 | 0.500 | 0.470 | 0.530 |
| recall@3 | 0.530 | 0.667 | 0.758 | 0.697 | 0.697 | 0.742 | 0.727 | 0.682 | 0.667 | 0.712 | 0.636 | 0.667 | 0.667 | 0.697 | 0.697 |
| recall@5 | 0.621 | 0.727 | 0.803 | 0.742 | 0.803 | 0.803 | 0.833 | 0.697 | 0.727 | 0.758 | 0.742 | 0.758 | 0.742 | 0.742 | 0.727 |
| recall@10 | 0.758 | 0.773 | 0.833 | 0.833 | 0.848 | 0.848 | 0.879 | 0.742 | 0.758 | 0.803 | 0.818 | 0.803 | 0.803 | 0.758 | 0.833 |
| hit@1 | 0.364 | 0.455 | 0.636 | 0.636 | 0.576 | 0.667 | 0.606 | 0.455 | 0.545 | 0.636 | 0.545 | 0.545 | 0.576 | 0.515 | 0.606 |
| hit@5 | 0.697 | 0.788 | 0.848 | 0.788 | 0.848 | 0.818 | 0.879 | 0.758 | 0.758 | 0.788 | 0.788 | 0.818 | 0.788 | 0.788 | 0.818 |
| hit@10 | 0.818 | 0.818 | 0.879 | 0.879 | 0.909 | 0.879 | 0.939 | 0.788 | 0.788 | 0.818 | 0.879 | 0.848 | 0.848 | 0.788 | 0.879 |
| precision@1 | 0.364 | 0.455 | 0.636 | 0.636 | 0.576 | 0.667 | 0.606 | 0.455 | 0.545 | 0.636 | 0.545 | 0.545 | 0.576 | 0.515 | 0.606 |
| precision@5 | 0.200 | 0.188 | 0.212 | 0.200 | 0.218 | 0.218 | 0.224 | 0.170 | 0.182 | 0.194 | 0.188 | 0.194 | 0.188 | 0.176 | 0.170 |
| precision@10 | 0.127 | 0.106 | 0.109 | 0.118 | 0.115 | 0.118 | 0.118 | 0.091 | 0.094 | 0.103 | 0.103 | 0.103 | 0.100 | 0.091 | 0.100 |
| context_precision | 0.459 | 0.579 | 0.716 | 0.690 | 0.667 | 0.688 | 0.699 | 0.586 | 0.641 | 0.680 | 0.652 | 0.645 | 0.674 | 0.638 | 0.664 |
| mrr@10 | 0.503 | 0.596 | 0.724 | 0.710 | 0.680 | 0.737 | 0.712 | 0.589 | 0.641 | 0.699 | 0.654 | 0.655 | 0.676 | 0.632 | 0.685 |
| ndcg@10 | 0.465 | 0.609 | 0.710 | 0.708 | 0.688 | 0.727 | 0.721 | 0.609 | 0.659 | 0.704 | 0.670 | 0.678 | 0.683 | 0.600 | 0.633 |
| map@10 | 0.362 | 0.538 | 0.657 | 0.652 | 0.620 | 0.666 | 0.653 | 0.548 | 0.618 | 0.662 | 0.610 | 0.622 | 0.633 | 0.543 | 0.557 |
| noise_ratio | 0.724 | 0.849 | 0.843 | 0.830 | 0.838 | 0.836 | 0.833 | 0.846 | 0.842 | 0.834 | 0.840 | 0.828 | 0.840 | 0.607 | 0.683 |
| redundancy | 0.493 | 0.332 | 0.337 | 0.353 | 0.345 | 0.338 | 0.348 | 0.310 | 0.338 | 0.351 | 0.350 | 0.327 | 0.347 | 0.396 | 0.449 |
| context_words | 985 | 1055 | 1072 | 1047 | 1067 | 1056 | 1051 | 1049 | 1049 | 1044 | 1048 | 1036 | 1043 | 908 | 841 |
| cross_doc_coverage | 0.000 | 0.400 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.400 | 0.600 | 0.600 | 0.400 | 0.600 | 0.400 | 0.400 | 0.400 |
| amend_coverage | 0.000 | 0.667 | 0.667 | 0.333 | 0.667 | 0.667 | 0.667 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |
| amend_only_superseded | 0.667 | 0.333 | 0.333 | 0.667 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.667 | 0.333 | 0.333 | 0.667 |

## Hypotheses (§11, judged on dev)

**H2** — `R-hybrid` beats `R-dense` on `lookup` questions that contain exact identifiers. Subset: 7 of 8 dev lookup questions match the identifier rule (`report.has_identifier`: digit, quoted term, section/clause reference, or a defined term = ≥ 2 capitalized words not at the start); unanswerable ones drop out of the metric. Test: paired recall@budget difference per chunker, no rerank. Rule: supported if every CI is above 0, not supported if any is below 0, else inconclusive.

| Comparison | n | recall@budget diff [95% CI] | CI vs 0 | mrr@10 diff [95% CI] (diagnostic) |
|---|---|---|---|---|
| `C2-hybrid` − `C2-dense` | 7 | +0.000 [-0.429, +0.429] | includes 0 | +0.016 [-0.060, +0.107] |
| `C4-hybrid` − `C4-dense` | 7 | +0.000 [+0.000, +0.000] | includes 0 | -0.048 [-0.357, +0.214] |

Verdict (dev): **inconclusive**.

**H3** — Reranking improves `mrr@10` for every base retriever. Test: paired mrr@10 difference, reranked − base, every base configuration in S2. Same rule.

| Comparison | n | mrr@10 diff [95% CI] | CI vs 0 |
|---|---|---|---|
| `C2-dense+rerank` − `C2-dense` | 33 | +0.127 [+0.010, +0.245] | above |
| `C2-bm25+rerank` − `C2-bm25` | 33 | -0.031 [-0.129, +0.065] | includes 0 |
| `C2-hybrid+rerank` − `C2-hybrid` | 33 | -0.026 [-0.130, +0.077] | includes 0 |
| `C4-dense+rerank` − `C4-dense` | 33 | +0.052 [-0.048, +0.154] | includes 0 |
| `C4-bm25+rerank` − `C4-bm25` | 33 | -0.045 [-0.130, +0.027] | includes 0 |
| `C4-hybrid+rerank` − `C4-hybrid` | 33 | +0.022 [-0.093, +0.138] | includes 0 |
| `C2-pc+rerank` − `C2-pc` | 33 | +0.052 [-0.114, +0.221] | includes 0 |

Verdict (dev): **inconclusive**.

## Retrieval latency by component (ms)

Per question, after the session index is built (index building is offline, as in production). Rerank runs
on CPU.

| Run | query_embed p50 / p95 | vector_search p50 / p95 | rerank p50 / p95 | bm25 p50 / p95 | fusion p50 / p95 | parent_map p50 / p95 | total p50 / p95 |
|---|---|---|---|---|---|---|---|
| `C1-dense` | 13.9 / 20.4 | 0.0 / 0.2 | – | – | – | – | 14.0 / 20.6 |
| `C2-dense` | 10.5 / 15.3 | 0.1 / 0.1 | – | – | – | – | 10.5 / 15.4 |
| `C2-dense+rerank` | 14.7 / 27.6 | 0.1 / 0.3 | 357.8 / 958.2 | – | – | – | 391.5 / 1038.9 |
| `C2-bm25` | – | – | – | 0.1 / 0.2 | – | – | 0.1 / 0.2 |
| `C2-bm25+rerank` | – | – | 329.6 / 495.6 | 0.2 / 0.4 | – | – | 343.6 / 514.3 |
| `C2-hybrid` | 12.0 / 14.1 | 0.1 / 0.1 | – | 0.2 / 0.2 | 0.0 / 0.0 | – | 12.2 / 14.5 |
| `C2-hybrid+rerank` | 13.3 / 17.7 | 0.1 / 0.3 | 303.5 / 733.4 | 0.2 / 0.4 | 0.0 / 0.1 | – | 327.7 / 770.3 |
| `C4-dense` | 9.4 / 11.3 | 0.0 / 0.1 | – | – | – | – | 9.5 / 11.5 |
| `C4-dense+rerank` | 13.4 / 24.6 | 0.1 / 0.5 | 424.9 / 936.6 | – | – | – | 452.7 / 983.2 |
| `C4-bm25` | – | – | – | 0.1 / 0.2 | – | – | 0.1 / 0.2 |
| `C4-bm25+rerank` | – | – | 381.2 / 717.0 | 0.2 / 0.4 | – | – | 392.6 / 744.8 |
| `C4-hybrid` | 9.8 / 16.2 | 0.1 / 0.2 | – | 0.2 / 0.2 | 0.0 / 0.1 | – | 10.0 / 16.6 |
| `C4-hybrid+rerank` | 14.0 / 32.7 | 0.1 / 0.4 | 563.9 / 1177.0 | 0.2 / 0.6 | 0.0 / 0.1 | – | 605.4 / 1214.5 |
| `C2-pc` | 11.2 / 19.4 | 0.1 / 0.4 | – | – | – | 2.4 / 12.6 | 16.5 / 31.6 |
| `C2-pc+rerank` | 12.1 / 16.2 | 0.1 / 0.4 | 870.2 / 1328.6 | – | – | 3.4 / 10.3 | 916.8 / 1415.8 |

## Rerank truncation

Candidates whose (question, chunk) pair exceeds the cross-encoder's 512 tokens are cut from the chunk's end before scoring.

| Run | candidates scored | truncated | share |
|---|---|---|---|
| `C2-dense+rerank` | 706 | 0 | 0.0% |
| `C2-bm25+rerank` | 704 | 0 | 0.0% |
| `C2-hybrid+rerank` | 706 | 0 | 0.0% |
| `C4-dense+rerank` | 732 | 0 | 0.0% |
| `C4-bm25+rerank` | 723 | 0 | 0.0% |
| `C4-hybrid+rerank` | 732 | 0 | 0.0% |
| `C2-pc+rerank` | 682 | 421 | 61.7% |

## Notes

- n = 33 answerable dev questions (7 unanswerable left out of coverage metrics). Type and part breakdowns have single-digit n: read them as diagnostics, not results.
- Cross-document coverage (n = 5) and amendment coverage (n = 3) are too small to compare chunkers; H9 is judged on the full set in R8.
- The test split is untouched: S1–S3 pick on dev only (§9.3).
- `R-pc` returns parents built per question, so its ndcg@10 / map@10 use the children's relevant count as the ideal: diagnostic only, not comparable with other retrievers. recall@budget, entity recall and mrr@10 don't depend on it (S-2026-11-22-44).
- Baseline `C1-dense` is the S1 run (same questions file, manifest, embedder, parse config and cleaner — checked); it is compared against, not picked.
- **Evidence fragmentation is 0 for every chunker.** QA evidence spans are short (about one sentence), so each span lies at least 50% inside a single chunk of every chunker. On this question set the metric does not separate chunkers.

## Provenance

Questions file sha256 `ddca7f3149032593…`, manifest `5daa8444816306f5…`, embedder `sentence-transformers/all-MiniLM-L6-v2|normalized|float32`, parse `P2` (scans `P2V`), cleaner `v2`.

| Run | git SHA | questions | cache | MLflow run | seconds |
|---|---|---|---|---|---|
| `C1-dense` | 938ea7b4005b | 40 | {'built': 44, 'loaded': 11} | 50051ca2bcf4 | 84.5 |
| `C2-dense` | a5b5f968867c | 40 | {'built': 0, 'loaded': 55} | 817ec598afdd | 1.0 |
| `C2-dense+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 80c951a09d7d | 17.5 |
| `C2-bm25` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | d02df2c58772 | 1.0 |
| `C2-bm25+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | b3732933561c | 16.1 |
| `C2-hybrid` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 9e9264fba94c | 1.5 |
| `C2-hybrid+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 41211baa484f | 16.0 |
| `C4-dense` | a5b5f968867c | 40 | {'built': 0, 'loaded': 55} | d79b9d731d1d | 0.8 |
| `C4-dense+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | a283b1d0be59 | 19.9 |
| `C4-bm25` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 71611063da26 | 1.0 |
| `C4-bm25+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 8a26db02a53c | 17.9 |
| `C4-hybrid` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 1605d63bda3c | 2.1 |
| `C4-hybrid+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 259aac1068c7 | 26.0 |
| `C2-pc` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | f66afce489d1 | 6.9 |
| `C2-pc+rerank` | a5b5f968867c | 40 | {'built': 0, 'loaded': 0} | 0517c2447045 | 39.3 |
