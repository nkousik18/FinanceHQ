# S3 Results — Index types (dev split)

**Generated** by `venv/bin/python -m evals.report --stage S3 --split dev` from
`data/results/r4/s3/dev/` (per-session runs: `venv/bin/python -m evals.run --config
evals/configs/s3_index_types.yaml`; pooled study: `venv/bin/python -m evals.index_study --config
evals/configs/s3_index_types.yaml`). Do not edit numbers by hand (F7). Parameters and scope:
`docs/18_session_log.md` (S-2026-11-22-33, S-2026-11-23-9).

**No pick** (§9.3): this stage shows the recall-vs-latency curve of approximate FAISS indexes against
exact search (`IndexFlatIP`, production). Configuration = the S2 winner, `C2` + `R-hybrid` + rerank; only
the dense list of R-hybrid uses the FAISS index. "Recall vs exact" (§8.3) = share of exact search's top k
that the approximate index's top k contains, per query, averaged.

## Per-question session indexes (as in every other stage)

Session index sizes over the 40 dev questions: 3–1489 chunks (median
563). Latency per question; the total is dominated by the CPU reranker.

| Run | Index | ★ recall@budget (n=33) | diff vs Flat [95% CI] | dense top-20 recall vs exact (mean) | vector search µs p50 / p95 | rerank ms p50 | total ms p50 |
|---|---|---|---|---|---|---|---|
| `flat` | Flat (exact) | **0.864** [0.742, 0.955] | – | 1 (exact) | 83 / 693 | 485 | 520 |
| `hnsw-ef16` | HNSW M=32 efC=200 efSearch=16 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.950 | 99 / 489 | 469 | 512 |
| `hnsw-ef32` | HNSW M=32 efC=200 efSearch=32 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.994 | 93 / 136 | 300 | 324 |
| `hnsw-ef64` | HNSW M=32 efC=200 efSearch=64 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.999 | 114 / 168 | 286 | 311 |
| `hnsw-ef128` | HNSW M=32 efC=200 efSearch=128 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.999 | 161 / 217 | 289 | 315 |
| `ivf-np1` | IVF nlist=sqrt nprobe=1 | **0.833** [0.712, 0.939] | -0.030 [-0.121, +0.061] | 0.428 | 36 / 46 | 292 | 319 |
| `ivf-np2` | IVF nlist=sqrt nprobe=2 | **0.803** [0.667, 0.924] | -0.061 [-0.152, +0.000] | 0.667 | 40 / 45 | 293 | 318 |
| `ivf-np4` | IVF nlist=sqrt nprobe=4 | **0.803** [0.667, 0.924] | -0.061 [-0.152, +0.000] | 0.836 | 47 / 61 | 306 | 329 |
| `ivf-np8` | IVF nlist=sqrt nprobe=8 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.929 | 58 / 90 | 313 | 340 |
| `ivf-np16` | IVF nlist=sqrt nprobe=16 | **0.864** [0.742, 0.955] | +0.000 [+0.000, +0.000] | 0.990 | 75 / 93 | 304 | 329 |

Runs execute in config order. Total latency is the CPU reranker's (rerank p50 286–485 ms across runs, scoring the same kind of candidates), so differences in the total column are reranker/CPU noise, not the index: compare the vector search column.

## Pooled index: the whole dev corpus

9,592 `C2` vectors (52 documents, each once, scans included;
d = 384), 40 dev questions as queries. Search latency = median of
5 single-query searches per question, then p50 / p95 over questions (FAISS
1.8.0, 1 thread — single-threaded in the evals, see `dense._faiss`). Top 20 is
what R-hybrid takes from the dense list.

| Index | build s | size MB | recall vs exact @10 | recall vs exact @20 | search µs p50 / p95 |
|---|---|---|---|---|---|
| Flat (exact) | 0.00 | 14.1 | 1.000 (min 1.00) | 1.000 (min 1.00) | 374 / 399 |
| HNSW M=32 efC=200 efSearch=16 | 2.10 | 16.5 | 0.900 (min 0.50) | 0.860 (min 0.50) | 22 / 27 |
| HNSW M=32 efC=200 efSearch=32 | 2.15 | 16.5 | 0.955 (min 0.70) | 0.944 (min 0.75) | 34 / 42 |
| HNSW M=32 efC=200 efSearch=64 | 2.09 | 16.5 | 0.975 (min 0.80) | 0.981 (min 0.90) | 58 / 71 |
| HNSW M=32 efC=200 efSearch=128 | 2.17 | 16.5 | 0.985 (min 0.80) | 0.990 (min 0.90) | 109 / 131 |
| IVF nlist=sqrt nprobe=1 (nlist used 98) | 0.04 | 14.3 | 0.410 (min 0.00) | 0.395 (min 0.00) | 13 / 16 |
| IVF nlist=sqrt nprobe=2 (nlist used 98) | 0.04 | 14.3 | 0.620 (min 0.00) | 0.590 (min 0.05) | 18 / 24 |
| IVF nlist=sqrt nprobe=4 (nlist used 98) | 0.04 | 14.3 | 0.750 (min 0.00) | 0.720 (min 0.05) | 27 / 35 |
| IVF nlist=sqrt nprobe=8 (nlist used 98) | 0.04 | 14.3 | 0.840 (min 0.00) | 0.820 (min 0.05) | 46 / 59 |
| IVF nlist=sqrt nprobe=16 (nlist used 98) | 0.04 | 14.3 | 0.930 (min 0.40) | 0.919 (min 0.35) | 86 / 101 |

## Notes

- Index building is offline (as production builds a session's index at upload); build time and size are
  shown for the rollout question (R8: does the winner fit a 1 GB host?).
- The test split is untouched (§9.3).

## Provenance

Questions file sha256 `ddca7f3149032593…`, manifest `5daa8444816306f5…`,
embedder `sentence-transformers/all-MiniLM-L6-v2|normalized|float32`. Pooled study git SHA `e5ecdd5926db`.

| Run | git SHA | MLflow run | seconds |
|---|---|---|---|
| `flat` | e5ecdd5926db | 23065b99030e | 25.2 |
| `hnsw-ef16` | e5ecdd5926db | 49642b71df96 | 25.8 |
| `hnsw-ef32` | e5ecdd5926db | a2f1103ee3f7 | 14.8 |
| `hnsw-ef64` | e5ecdd5926db | 4bc92df676ad | 13.2 |
| `hnsw-ef128` | e5ecdd5926db | 57310f9a74ba | 13.5 |
| `ivf-np1` | e5ecdd5926db | 887a8ee5786b | 12.7 |
| `ivf-np2` | e5ecdd5926db | a56d73bcf86c | 12.8 |
| `ivf-np4` | e5ecdd5926db | 445c8164373c | 14.6 |
| `ivf-np8` | e5ecdd5926db | 4d3b4623df0e | 14.3 |
| `ivf-np16` | e5ecdd5926db | 3218df35c0b0 | 13.0 |
