# Session Log

Dated record of working sessions: what changed, what was decided, and why. Newest first.
Add a new `## YYYY-MM-DD` section per session. Decisions get an ID (`S-<date>-<n>`) so later docs can cite them.

Related: initiative scope `loandoc-retrieval-evals-scope-v2.md` (its own Decision Log holds the D10 implementation choices) · R0 details `docs/r0_findings.md`.

---

## 2026-11-24 — R6 step 1: generation harness, test-split retrieval, smoke run

**Branch:** `feature/r6-generation`. Kousik's answer to the R6 proposal (S-2026-11-23-26): the Groq free tier is **one daily token budget per account** shared by all models, input and output, so the 15-day plan (~2.85M tokens) is out. Priority: **resume numbers first, only what's necessary, a fully working pipeline end to end**; the rest later. Step 1 approved.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-24-1 | **Groq limit = one budget per account (~200k tokens/day, all models).** The "per model → ~8 days" option is rejected | Kousik checked; ledger's single 190k guard was already right | ledger unchanged |
| S-2026-11-24-2 | **Dev extras (noise sensitivity, 3-run consistency, ~680k tokens) deferred**, not cut from scope. Options offered: keep in full; 2 runs instead of 3; only 13 calc + 7 unanswerable — all set aside for now | Biggest optional cost; not needed for the headline S4 numbers | this log |
| S-2026-11-24-3 | **Harness** `evals/generation/` (`prompt.py`, `stream.py`, `metrics.py`, `run.py`) + `evals/configs/s4_generation.yaml`; generator = production settings (`qwen/qwen3.8-27b`, temp 0.1, max 1,024, single user message) | §8.4, §8.5, §9.5, D5 | code |
| S-2026-11-24-4 | **Context = the retrieval run's `in_budget` chunks** rebuilt from the R4 chunk cache (production `_build_context` rule via `budget_cut`), not re-retrieved | One retrieval per config; generation sees exactly what retrieval was scored on | `run.py` docstring |
| S-2026-11-24-5 | **Chunks labelled `S1…Sn` in the prompt**, mapped back to chunk ids for the citation metrics. Rejected: raw chunk ids (`deal_10_am1-C2-75`) | Fewer prompt tokens and fewer mangled ids; metrics stay per chunk as D5 asks | `prompt.py` docstring |
| S-2026-11-24-6 | **One citation template (`cite-v1`) for every question type.** Rejected: production's intent-routed templates + A/B selector | Configurations must differ only in retrieval. Fixed strings: refusal = "Not found in the document." (production lookup phrase); calculations end "Final answer: …" | `prompt.py` |
| S-2026-11-24-7 | **Metric rules:** refusal = the exact phrase anywhere or a decline pattern in the *first* sentence; false refusal counted only when recall@budget = 1 (partial evidence may justify a refusal); stale = states a superseded value and none of the current entities; an unknown cited id counts as a wrong citation; numeric EM reads the "Final answer:" line, else the last number (million/billion scaled) | Deterministic, judge-free (§8.4); edge cases pinned by tests | `metrics.py` docstring, `test_generation.py` |
| S-2026-11-24-8 | **Added `answer_entity_recall`** (question entities found in the answer; refusal = 0) as a judge-free correctness proxy | Gives a correctness number before any judge spend; limits stated in the docstring (ignores wrong extra claims) | `metrics.py` |
| S-2026-11-24-9 | **Runner order: question by question, every configuration per question** | A run stopped by the daily budget still leaves paired results | `run.py` |
| S-2026-11-24-10 | **qwen/qwen3.8-27b price $0.80 in / $4.00 out per M** (Groq preview list price via price trackers; not yet confirmed on groq.com/pricing) | D7 needs a price even on the free tier | `ledger.py` |
| S-2026-11-24-11 | **Retrieval on test (local, 0 tokens), runner log:** recall@budget C2-bm25 0.846, C2-hybrid+rerank 0.811, C2-bm25+rerank 0.771, C1-dense 0.585. Not quotable until a generated report covers it (F7); test ranking ≠ dev ranking (dev: hybrid+rerank first) | S4 input | `data/results/r4/s{1,2}/test/` |
| S-2026-11-24-12 | **Smoke (dev q0004–q0006 × C2-hybrid+rerank, C1-dense): 6 calls, 11,412 tokens, $0.0096.** Prompt 1,670–2,224 tokens, completion 7–56, **no reasoning tokens**, TTFT 0.52–0.89 s, e2e ≤ 0.92 s. Hybrid+rerank answered 3/3 with correct citations; C1-dense refused 2/3 — both where its recall@budget was 0 | Replaces the assumed 2,050 tokens/call | `data/results/r6/s4/dev/` |
| S-2026-11-24-13 | **Revised estimate:** generation ≈ 1,920 tokens/call (measured). Full S4 generation 320 calls ≈ 615k (~3.2 days); dev-picked winner + baseline only, 160 calls ≈ 307k (~1.6 days). Judge still assumed ≈ 2,800/call (to measure) | — | — |
| S-2026-11-24-14 | Tests: evals **458** pass (+24 in `test_generation.py`, fake stream, no network) | N3 | — |

---

## 2026-11-23 (cont.) — R6 start: plan proposal and token estimate

**Branch:** `feature/r6-generation` (from `main` at `a5771b6`, after PR #9 merged). Kousik: "start r6".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-23-24 | **PR #9 (R5) merged to `main`** by Kousik, 2026-11-23 15:22 UTC (merge commit `a5771b6`) | — | GitHub #9 |
| S-2026-11-23-25 | **Token estimate inputs (measured):** mean context words in the S2/S1 runs: C2-hybrid+rerank 1,051, C2-bm25 1,047, C2-bm25+rerank 1,067, C1-dense 985 (max 1,200). **Assumed** (to be replaced by smoke-run numbers): 1.35 tokens/word; generator ≈ 1,850 in + 200 out ≈ 2,050 tokens/call; judge ≈ 2,600 in + 900 out (incl. hidden reasoning) ≈ 3,500 tokens/call | Estimate before any LLM run (Kousik's rule) | proposal below |
| S-2026-11-23-26 | **Estimate (proposed, pending Kousik):** smoke ~20k; calibration 50 gen + 50 judge ≈ 280k (+ ~90k if a 25-answer re-test is needed); S4 320 gen + 320 judge ≈ 1.78M; dev extras (top config: noise sensitivity 2 × 40 gen+judge, consistency 3 × 40 gen) ≈ 680k. **Total ≈ 2.85M tokens ≈ 15 days at 190k/day if the limit is per account**; $ (gpt-oss-120b list price) ≈ $0.30 for S4 judging — far below D7 caps | — | — |

---

## 2026-11-23 (cont.) — R5 wrap-up: PR, options considered, logging rule

Kousik asked that **every discovery and every discussion** be logged, not only final decisions, so that each decision can be checked or defended later. This section adds what earlier sections left out: the options presented to Kousik and the ones rejected, with reasons.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-23-15 | **R5 PR #9 opened** (`feature/r5-retrieval-strategies` → `main`), waiting for Kousik's review. R6 not started (needs Kousik's go) | Plan S-2026-11-22-31 ends with the PR | GitHub #9 |
| S-2026-11-23-16 | **Rejected (S2 R-pc variants, 2026-11-22):** a second R-pc configuration with C3 children → C4 parents (16 runs). Rejected on the agent's recommendation: C3 lost S1 (0.652 vs C2 0.773), and the 2 extra runs would test a chunker already ruled out | Keeps S2 at 14 runs, all on S1's picks | S-2026-11-22-32 |
| S-2026-11-23-17 | **Rejected (S3 scope, 2026-11-22):** per-session indexes only. Rejected because sessions hold 3–1,489 chunks, too few for HNSW/IVF to differ from Flat; the S3 results confirmed it (per-session recall vs exact ≥ 0.95 for every HNSW setting) | The curve needs a bigger index | S-2026-11-22-33, S-2026-11-23-12 |
| S-2026-11-23-18 | **Rejected (R-pc + rerank, 2026-11-23, Kousik chose option 1):** reranking R-pc's *children* before parent mapping (option 2). It would avoid the 62% truncation of ~800-token parents, but it changes §7's "rerank the candidates" and would need a Decision Log deviation. Result for the record: R-pc + rerank is the weakest S2 config (0.667), and the truncation is a likely contributor; not tested | Follow the spec; measure the cost | S-2026-11-22-49, S-2026-11-23-4 |
| S-2026-11-23-19 | **Rejected (S2 picks, 2026-11-23, Kousik kept the rule):** replacing the third pick C2-bm25+rerank with C2-dense+rerank for a more varied S4. Both have recall@budget 0.833; the rule's choice rests on a ~50 ms rerank-latency gap (CPU noise) | Pre-committed rule over a post-hoc choice | S-2026-11-23-8 |
| S-2026-11-23-20 | **Rejected (R-pc design, agent, 2026-11-22):** fixed, non-overlapping 800-token blocks per section as parents (simpler duplicate removal, parents precomputable). Rejected for centred windows (as proposed to Kousik), so a child near a block edge still gets context on both sides; overlap handled by the covered-child skip (S-2026-11-22-43) | Matches the approved proposal | S-2026-11-22-42 |
| S-2026-11-23-21 | **Rejected (R-pc rank metrics, agent):** counting relevant *parents* for ndcg/map (map every relevant child to its parent). Rejected as itself an approximation (a non-relevant child's parent can also cover evidence); children count kept and the caveat printed in the report | Decision metric unaffected | S-2026-11-22-44 |
| S-2026-11-23-22 | **Rejected (BM25 zero-score results, agent):** returning zero-score chunks in index order to always fill k. Rejected: their rank would be arbitrary. Consequence: R-bm25 can return fewer than 20 chunks (e.g. 704 rerank candidates for 40 questions vs 706 for dense) | Honest lower bound | S-2026-11-22-37 |
| S-2026-11-23-23 | **Logging rule (Kousik):** from now on each session section records findings (with numbers), every option put to Kousik **and the ones rejected, with reasons**, smoke-check observations, bugs found, and PR/merge events — not only final decisions | So decisions can be proved or revisited from the log | memory + this log |

---

## 2026-11-23 (cont.) — R5 step 5: S3 index-type study

**Branch:** `feature/r5-retrieval-strategies`. Kousik: "Start step 5 by keeping the rules pick". Runs at `e5ecdd5` (clean tree), dev. Report: `docs/24_s3_index_types.md` (no pick); MLflow `loandoc_retrieval_evals`: 10 per-session runs + 10 pooled runs (tag `study=pooled`).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-23-9 | **Index parameters (D10):** HNSW M 32, efConstruction 200, efSearch ∈ {16, 32, 64, 128}; IVF nlist = round(√N) per index (IVFFlat, trained on the index's own vectors), nprobe ∈ {1, 2, 4, 8, 16}; all inner product on normalized MiniLM vectors. Config `evals/configs/s3_index_types.yaml`; spec in a run's `index:` (`evals/retrievers/ann.py`) | Standard sweep points; √N is the usual IVF starting point | Decision Log |
| S-2026-11-23-10 | **Recall vs exact (§8.3)** = share of exact search's top k inside the approximate top k. Per session: a non-flat index keeps an exact IndexFlatIP copy; the dense list's overlap is computed outside the timers (`exact_overlap` in `retriever_info`). Pooled: `evals/index_study.py`, one index over the 52 dev-session documents (each once, no dropped pages; 9,592 C2 vectors), 40 dev questions, top 10/20, latency = median of 5 single-query searches per question | §8.3; S-2026-11-22-33 | — |
| S-2026-11-23-11 | **Bug found: FAISS batch search (nq > 1) segfaults when torch is loaded** (two OpenMP runtimes, macOS; single-query search was safe, which is all the runner does). **`dense._faiss()` now sets FAISS to 1 thread** for all evals; this also makes HNSW/IVF builds reproducible. Flat S3 run reproduces S2's C2-hybrid+rerank exactly (0.864) | Same family as S-2026-11-22-15; a small CPU host is single-threaded anyway | `evals/retrievers/dense.py` |
| S-2026-11-23-12 | **S3 per-session result:** HNSW (every efSearch) and IVF nprobe 8/16 give the **same recall@budget as Flat (0.864)**; IVF nprobe 1/2/4: 0.833 / 0.803 / 0.803 (diff vs Flat CIs touch 0). Dense top-20 recall vs exact: HNSW 0.950 → 0.999 (ef 16 → 128), IVF 0.428 → 0.990 (np 1 → 16). Vector search is 36–161 µs p50 on sessions of 3–1,489 chunks: **irrelevant next to query embedding (~12 ms) and CPU rerank (~300 ms)** | §9.3 S3 | — |
| S-2026-11-23-13 | **S3 pooled curve (9,592 vectors):** Flat 374 µs p50, recall 1.0; HNSW ef 16/32/64/128 → recall@20 0.860 / 0.944 / 0.981 / 0.990 at 22 / 34 / 58 / 109 µs (build 2.1 s, 16.5 MB vs 14.1 MB); IVF np 1→16 → recall@20 0.395 → 0.919 at 13 → 86 µs (build 0.04 s). **At this corpus size Flat stays the right choice** (exact, < 0.4 ms); HNSW ef 64 would be the pick only at far larger N | §7 "the point is the curve" | R8 |
| S-2026-11-23-14 | Per-session total latency: the first two runs (flat, hnsw-ef16) were ~520 ms p50 vs ~320 ms for the rest — **reranker CPU warm-up** (rerank p50 485 / 469 vs 286–304 ms), not the index. Report adds a rerank column and a generated note | Don't misread order effects as index effects | `evals/report.py` |

Changes: `evals/retrievers/{ann,dense,hybrid}.py`, `evals/run.py` (`index`), `evals/index_study.py`, `evals/report.py` (`render_s3`), `evals/configs/s3_index_types.yaml`, `evals/tests/test_index_types.py` (14 tests). Evals 434 pass. `docs/24_s3_index_types.md` (generated); S1/S2 reports regenerate byte-identical.

**R5 complete:** R-bm25, R-hybrid, R-pc, +rerank, S2 (picks C2-hybrid+rerank, C2-bm25, C2-bm25+rerank + C1-dense baseline → S4), S3 (Flat stays). Next: R5 PR → Kousik's review; R6 (generation, judge, S4 on test) only when Kousik says go.

---

## 2026-11-23 — R5 step 4: S2 run + report

**Branch:** `feature/r5-retrieval-strategies`. Kousik: "option 1, go step 4" (R-pc + rerank reranks the parents as specified; truncation reported — closes S-2026-11-22-49). Run at `a5b5f96` (clean tree), dev 40 q (33 answerable), top_k 20. Report: `docs/23_s2_retrieval.md`; MLflow `loandoc_retrieval_evals`, 14 runs.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-23-1 | **S2 config** `evals/configs/s2_retrievers.yaml`: 14 runs (C2, C4 × dense/bm25/hybrid ± rerank; C2→C4 R-pc ± rerank). **Baseline = S1's C1-dense run**, named in the config (`baseline: {stage, run}`), loaded first by the report, **refused unless** questions file, manifest, embedder, parse config and cleaner all match; not a pick candidate | §9.3 "plus the C1 + R-dense baseline"; no need to re-run it | `evals/report.py` |
| S-2026-11-23-2 | **H2 subset rule fixed before the run** (`report.has_identifier`): a lookup question "contains exact identifiers" if it has a digit, a quoted term, a section/clause reference, or ≥ 2 capitalized words not at the start (defined term). 7 / 8 dev lookup questions match. Test: paired recall@budget hybrid − dense per chunker, no rerank. **Hypothesis verdict rule** (H2, H3): supported = every CI above 0; not supported = any CI below 0; else inconclusive | §11 gives no operational definition | Decision Log |
| S-2026-11-23-3 | **Bug found in the first S2 run and fixed before reporting:** BM25's index was built lazily on each session's first question, inside the runner's `total` timer (R-bm25 total p95 ≈ 81 ms vs 0.3 ms of BM25 work). Added a retriever `prepare(index)` hook called at session build; re-ran all 14 runs at `a5b5f96`. Quality metrics identical; the 14 first-attempt MLflow runs (`ec45b80`) tagged `superseded` | Latency must exclude index building (offline in production) | `evals/run.py` |
| S-2026-11-23-4 | **S2 result (recall@budget, 95% CI, n = 33):** C2-hybrid+rerank **0.864** [0.742, 0.955]; C2-bm25 / C2-bm25+rerank / C2-dense+rerank 0.833; C2-hybrid, C4-hybrid 0.818; C4-bm25(+rerank), C4-hybrid+rerank 0.803; C2-dense 0.773 (= S1); C4-dense+rerank 0.773; C4-dense 0.758; C2-pc 0.727; C2-pc+rerank 0.667; baseline C1-dense 0.515. **Every config except C2-pc+rerank beats the baseline** (CIs exclude 0); **no two neighbours in the ranking are distinguishable** | Decision metric (D3) |
| S-2026-11-23-5 | **S2 picks (top 3, D3 + ties → faster): C2-hybrid+rerank, C2-bm25, C2-bm25+rerank.** 3-way tie at 0.833 for slots 2–3 (C2-bm25, C2-bm25+rerank, C2-dense+rerank) decided by latency p50 (0.1 / 344 / 392 ms). The bm25+rerank vs dense+rerank order rests on CPU rerank-time noise (same order in both S2 attempts: 362 vs 409, 344 vs 392 ms). Two of the three picks share a base (C2-bm25) — **flagged to Kousik** | Rule as written; low diversity for S4 |
| S-2026-11-23-6 | **H2 (dev): inconclusive** — hybrid − dense on 7 identifier lookups: C2 +0.000 [−0.429, +0.429], C4 +0.000 (identical per question). **H3 (dev): inconclusive** — rerank improves mrr@10 only for C2-dense (+0.127 [+0.010, +0.245]); for BM25 bases the point estimate is negative (C2 −0.031, C4 −0.045), all other CIs include 0 | §11 |
| S-2026-11-23-7 | Findings for R8: **BM25 alone matches the best dense setups at ~0.1 ms and no query embedding**; R-pc is the weakest strategy (parents are ~800 tokens, so ≈ 3 fit the budget; noise ratio lowest but recall lower); reranking costs ≈ 300–560 ms p50 on CPU (870 ms on R-pc parents, 62% of which are truncated at 512 tokens); C4-hybrid+rerank rerank p50 varied 432 → 564 ms between the two attempts (CPU noise) | Rollout cost/benefit |
| S-2026-11-23-8 | **S2 picks kept as the rule gives (Kousik): C2-hybrid+rerank, C2-bm25, C2-bm25+rerank** → S4 (with the C1-dense baseline). Closes the flag in S-2026-11-23-5. S3 winner = C2-hybrid+rerank | Pre-committed rule over a post-hoc override | §9.3 |

Changes: `evals/configs/s2_retrievers.yaml`, `evals/report.py` (baseline, H2/H3, latency + truncation tables), `evals/run.py` (`baseline`, `prepare`), `evals/retrievers/{bm25,hybrid,rerank,parent_child}.py` (prepare; section tokens at build; reranker loaded at init), tests (evals 420 pass), `docs/23_s2_retrieval.md` (generated). S1 report regenerates byte-identical.

---

## 2026-11-22 (cont.) — R5 step 3: +rerank

**Branch:** `feature/r5-retrieval-strategies`. Kousik: "step 3".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-47 | **`+rerank`** (`evals/retrievers/rerank.py`): wraps any retriever; base top max(20, k) candidates → `cross-encoder/ms-marco-MiniLM-L-6-v2` scores (question, chunk text) → top k; ties → base order. **CPU** (production-like small host), max_length 512, batch = one question's candidates | §7 |
| S-2026-11-22-48 | Run config `rerank: true|false`; `make_retriever(name, rerank)`; summary + MLflow get `retriever_params.rerank`, tag `rerank`, latency component `rerank`, and per-run totals `retriever_info` (`rerank_candidates`, `rerank_truncated`; per question in `questions.jsonl`) | Per-component latency (§8.3); truncation visible | `evals/run.py`, `evals/tracking.py` |
| S-2026-11-22-49 | **Finding (smoke, dev, not S2): the reranker truncates 421 / 682 (62%) of R-pc parent candidates** at 512 tokens — it scores only the first ~490 tokens of an ~800-token parent. C2 / C4 candidates: 0 truncated. Rerank adds ≈ 170–220 ms p50 per question on CPU for 254-token chunks, ≈ 470 ms for parents (smoke run timings) | Affects how "R-pc + rerank" should be read. **Open for Kousik:** keep as specified (rerank the parents, report truncation) or rerank R-pc's children before parent mapping | — |

Changes: `evals/retrievers/{rerank,dense,__init__}.py`, `evals/run.py`, `evals/tracking.py`, `evals/tests/test_retriever_rerank.py` (9 tests, one with the real cross-encoder). Evals 407 pass. Smoke output deleted.

---

## 2026-11-22 (cont.) — R5 step 2: R-pc (parent–child)

**Branch:** `feature/r5-retrieval-strategies`. Kousik: "run step 2".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-41 | **R-pc parents** (`evals/retrievers/parent_child.py`): children = the run's chunker (C2), dense search; a C4 *section* = consecutive C4 chunks with the same heading path; parent text rebuilt from the cleaned page text under the union of the section's spans (C4's own chunks overlap and repeat the prefix), prefixed with the heading path → parent spans match parent text exactly. Child → the section sharing the most characters (ties → earlier) | §7; C4 chunks are ≤ 254 tokens, so "the C4 section" must be rebuilt from its chunks |
| S-2026-11-22-42 | **Window cap (D10): section > 800 MiniLM content tokens → 800-token window centred on the child**, clipped at the section ends (stays 800). **No C4 section** (no overlap, or empty heading path) → same window over the whole document | Unbounded sections (definitions run to 50+ pages) would fill the 1,200-word budget alone | Decision Log |
| S-2026-11-22-43 | **Dedupe:** a child is skipped if its parent is already returned **or ≥ 50% of its characters are inside a returned parent** (§8.3 coverage rule); parent score = best child's score; stops at k parents | Centred windows of nearby children overlap; without this they repeat text in the budget |
| S-2026-11-22-44 | Rank metrics' ideal (`n_relevant`, ndcg/map) stays the **children** count for R-pc: parents are built per query, so there's no fixed parent index. ndcg@10/map@10 for R-pc are diagnostic only; recall@budget and entity recall don't use it | Report caveat |
| S-2026-11-22-45 | Runner: `parent_chunker` in the run config (required for R-pc, rejected otherwise); parent C4 chunks come from the same chunk cache; `Hits.chunks` carries chunks not in the index; latency component `parent_map` | — | `evals/run.py` |
| S-2026-11-22-46 | **Checks:** provenance on real data — 19 dev views, 434 sampled parents, 0 spans whose text isn't in the parent text; parent size p50 811 / p95 822 / max 843 MiniLM tokens (800 + heading prefix). Smoke run C2-pc (dev, no MLflow): runs end to end, ~half of returned parents are windows, ≈ 2.9 parents fit the budget; output deleted, numbers come from S2 (F7) | Plumbing check |

Changes: `evals/retrievers/{parent_child,dense,__init__}.py`, `evals/run.py`, `evals/tests/test_retriever_pc.py` (10 tests). Evals 398 pass.

---

## 2026-11-22 (cont.) — R5 steps 0–1: branch, R-bm25, R-hybrid

**Branch:** `feature/r5-retrieval-strategies` (from `main` after PR #8 merged, `2482e45`). Kousik: "yes go, propose R5 steps", then "go with your recommendations, start step 0 and 1".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-31 | **R5 steps:** 0 branch + housekeeping → 1 `R-bm25` + `R-hybrid` → 2 `R-pc` → 3 `+rerank` → 4 S2 config + run + report (H2, H3) → 5 S3 index study → PR | Scope §12 R5; one module at a time | this log |
| S-2026-11-22-32 | **One `R-pc` configuration in S2: children C2 → parents C4 sections** (Kousik, on the agent's recommendation). "C4 × R-pc" would be the same run, since C4 chunks are the parents. **S2 = 14 runs** (C2, C4 × dense/bm25/hybrid = 6, + R-pc = 1; each ± rerank) ≤ 16 | S1 picked C2 + C4; §7 says children are C2/C3 settings, parents C4 sections | §9.3 S2 |
| S-2026-11-22-33 | **S3 runs on the per-question views as specified, plus one pooled index of the whole dev corpus** (Kousik, on recommendation): a single session's index is small enough that HNSW/IVF barely differ from Flat; the pooled index shows the curve | §7 "the point is the curve" | step 5 |
| S-2026-11-22-34 | Handoff (`docs/19_handoff.md`) refresh moved from the merged R4 branch to this branch; R4 marked merged | Was uncommitted on a merged branch | handoff §3 |
| S-2026-11-22-35 | **BM25 library: `bm25s` 0.3.16** (D10): in-process, pip, numpy only; Lucene BM25, k1 1.5, b 0.75. Added to `requirements-dev.txt` | §14 D10 needs in-process + pip; `rank_bm25` is slower and unmaintained | Decision Log |
| S-2026-11-22-36 | **BM25 tokenizer** (D10): lowercase, words + numbers, bm25s English stopwords, **no stemming**; numbers normalized (thousands commas dropped, trailing decimal zeros dropped: "$62,500.00" = "62500", "3.50" = "3.5"); "$"/"%" not tokens; "2.11" one token | H2 is about exact identifiers; stemming/splitting would blur them | Decision Log |
| S-2026-11-22-37 | **R-bm25 returns only chunks with score > 0** (may return fewer than k) | A zero-score chunk's rank is arbitrary | `evals/retrievers/bm25.py` |
| S-2026-11-22-38 | **R-hybrid:** dense + BM25 lists each max(20, k) deep (= §7's 20 at top_k 20), RRF k = 60 on ranks; ties → better best rank, then dense before BM25, then chunk id | §7; deterministic | `evals/retrievers/hybrid.py` |
| S-2026-11-22-39 | **Retriever interface:** `search(index, query_vec, k, query=...)` + `needs_vector`; registry moved to `evals/retrievers/__init__.py`. Runner skips the query embedding for R-bm25 (its latency has no embed time); latency components are whatever the retriever reports (`bm25`, `fusion` added); retriever params written to `summary.json` and MLflow (`retriever.*`) | §8.3 per-component latency; provenance | `evals/run.py`, `evals/tracking.py` |
| S-2026-11-22-40 | **Smoke check (not S2):** C2 × {dense, bm25, hybrid}, dev, top_k 20, no MLflow: all run end to end; C2-dense reproduces S1's recall@budget exactly (0.773), so the interface change didn't move the baseline. Output deleted; S2 numbers come from the step 4 report (F7) | Plumbing check | — |

Changes: `evals/retrievers/{__init__,bm25,hybrid,dense}.py`, `evals/run.py`, `evals/tracking.py`, `evals/tests/test_retrievers_lexical.py` (24 tests), `requirements-dev.txt`. Evals 388 pass.

---

## 2026-11-22 (cont.) — R4 step 5: S1 run

**Branch:** `feature/r4-retrieval-metrics`. Kousik: "go step 5". Run at `938ea7b` (clean tree), dev 40 q (33 answerable), top_k 20, no LLM. Report: `docs/22_s1_retrieval.md`; MLflow `loandoc_retrieval_evals`, 5 runs.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-24 | **S1 result (recall@budget, 95% CI):** C1 0.515 [0.364, 0.667] · C2 **0.773** [0.636, 0.894] · C3 0.652 · C4 0.758 · C5 0.758. vs baseline C1: **C2 +0.258** [+0.121, +0.409], **C4 +0.242**, **C5 +0.242** (Holm p ≤ 0.012), C3 +0.136 (CI touches 0: no detectable difference). C2 vs C4 vs C5: **no detectable difference** (n = 33). Entity recall same picture (C2/C4/C5 ≈ 0.80 vs C1 0.615) | Decision metric (D3) |
| S-2026-11-22-25 | **Pick: C2 (rank 1), then a tie C4 = C5 at 0.758** for the second slot. Report now detects ties at the pick boundary and says the order is config order only. Tie rule open (Kousik) | D3 has no tie rule for S1 |
| S-2026-11-22-26 | **Evidence fragmentation = 0 for all five chunkers** — checked independently: every evidence span (median 11 words, max 153) is ≥ 89% inside one chunk even for C3/C2. Real, not a bug; the metric doesn't separate chunkers on this set. Report prints this note (generated) | §8.2 ★ metric, first measurement |
| S-2026-11-22-27 | Diagnostics: C1 recall@10 0.758 vs recall@budget 0.515 — its big (truncated) chunks fill the 1,200 words with fewer chunks; C1 redundancy highest (0.49). Amendment questions: C1 never put the amending span in the budget (0/3; 2/3 only the superseded original); C2/C5 2/3. n too small, H9 in R8 | For the R8 write-up |
| S-2026-11-22-28 | **Bug fixed:** local `summary.json` lacked the MLflow run id (written before logging); now rewritten after logging (+ test); the 5 S1 summaries backfilled from MLflow by run name | Report provenance column showed "–" |
| S-2026-11-22-30 | **Tie rule (Kousik): faster wins**, as S0's "ties → faster"; faster = lower retrieval latency p50 (C4 9.3 ms vs C5 11.5 ms; C4 also indexes cold in 176 s vs 392 s). Encoded in `report.rank_order`. **S1 pick: C2 + C4 → S2.** Closes S-2026-11-22-25 | Pick follows a rule, not config order |
| S-2026-11-22-29 | Cost/time: cold indexing C1 85 s, C2 165 s, C4 176 s, C5 392 s (C3 cached); retrieval p50 8–14 ms (query embedding dominates) | — |

Changes: `evals/run.py`, `evals/report.py` (tie flag, generated notes), tests (evals 364 pass), `docs/22_s1_retrieval.md` (generated).

---

## 2026-11-22 (cont.) — R4 step 4: report generator

**Branch:** `feature/r4-retrieval-metrics`. Kousik: "go step 4".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-21 | **`evals/report.py`** (`--stage S1 --split dev` → `docs/22_s1_retrieval.md`, F7: generated only). ★ recall@budget and entity recall: mean + question-bootstrap 95% CI; paired per-question bootstrap vs the baseline (first config = C1 + R-dense, production) **and between neighbours in the ranking** (is #2 distinguishable from #3); Holm p vs baseline shown for information (§9.6 requires Holm in S4 only); pick = top 2 by recall@budget point estimate (D3), with the tie caveat printed; breakdowns by type and part; all other metrics diagnostic | §9.6, §9.3 |
| S-2026-11-22-22 | Report **refuses** runs on different question files / qid sets, smoke (`--limit`) runs, and missing runs | No mixed or partial comparisons |
| S-2026-11-22-23 | **Evidence fragmentation (★ §8.2) now measured in the runner**: per question, located spans no single chunk in its index covers (`metrics.index_coverage`); report pools it with a question-bootstrap ratio CI. Closes R2's "pending QA set" | R2 left it for R3/R4 |

Changes: `evals/report.py`, `evals/retrieval/metrics.py` (`index_coverage`), `evals/run.py` (`spans`, `fragmented` per row), `evals/tests/test_report.py` (2), 1 metrics test. Evals 363 pass; app 173 pass, 2 skipped. The C3 check run from step 2 predates the fragmentation field: S1 (step 5) re-runs everything.

---

## 2026-11-22 (cont.) — R4 step 3: local MLflow logging

**Branch:** `feature/r4-retrieval-metrics`. Kousik: "go step 3 and s1 keep 20".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-18 | **S1 `top_k` = 20** (Kousik; closes S-2026-11-22-17): the 1,200-word budget binds, not the chunk count | Small-chunk chunkers not capped below the budget | `evals/configs/s1_chunkers.yaml` |
| S-2026-11-22-19 | **`evals/tracking.py`**: shared local MLflow (SQLite `data/mlflow/mlflow.db`, artifacts `data/mlflow/artifacts/`); S0's `log_mlflow` now uses it. Runner logs **one MLflow run per configuration** to `loandoc_retrieval_evals` by default (`--no-mlflow` to skip). Params: every run-config field, stage/split/limit, git SHA, manifest + questions sha256, embedder, table config, cleaning version, parser name/config/versions per downstream config (P2, P2V). Metrics: every breakdown group's means + n answerable, latency p50/p95 per component, unlocated spans. Artifacts: summary.json, questions.jsonl. `--limit` runs named/tagged **SMOKE** | §9.2 |
| S-2026-11-22-20 | Metric names: `@` → `_at_`, `type=lookup` → `type.lookup` (MLflow rejects `@` and `=`) | MLflow name rules |

Checked: a 2-question smoke run logged to the real store (43 params, 75 metrics, 2 artifacts). Changes: `evals/tracking.py`, `evals/run.py`, `evals/ingest_metrics/run_s0.py`, `evals/configs/s1_chunkers.yaml`, 3 tests. Evals 360 pass.

---

## 2026-11-22 (cont.) — R4 step 2: runner, chunk/embedding cache, R-dense

**Branch:** `feature/r4-retrieval-metrics`. Kousik: "start step 2".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-11 | **Scan views per question** (`evals/retrieval/views.py`): default index = the session's digital docs (scans left out); a question citing a scan gets the scan added and the scan's source page **dropped** from its source doc (page removed before cleaning: text, tables, form fields, layout elements). Swapping every scan in permanently was rejected: 15 digital questions cite pages that a scan copies | Decision Log 2026-11-13 left "scan in place of its source page" to R4 |
| S-2026-11-22-12 | **Chunk/embedding cache** (`evals/retrieval/index_cache.py`, `data/cache/chunks/`): one entry per document view × chunker; key = sha256(chunker_id, params, parser cache key, cleaning version, embedder, dropped pages, format). Entry stores cleaned page text + chunks (gz JSON) + vectors (npy), so cached runs neither re-parse nor re-clean; written json-last (no half entries); vector/chunk count mismatch → rebuild | §9.2 key + what scoring needs |
| S-2026-11-22-13 | **R-dense** (`evals/retrievers/dense.py`): FAISS IndexFlatIP per question view (production builds one per session), MiniLM normalized query, no threshold | §7 baseline = production |
| S-2026-11-22-14 | **Runner** `python -m evals.run --config … --split dev [--runs …] [--limit N]`: explicit run list in YAML (`evals/configs/s1_chunkers.yaml`, C1–C5 × R-dense); ranks max(top_k, 10) chunks, only the first top_k can enter the 1,200-word budget; ndcg/map ideal from relevant chunks in the whole view index. Output `data/results/r4/<stage>/<split>/<run>/questions.jsonl` + `summary.json` (git SHA, manifest + questions hashes, breakdowns, latency p50/p95, cache stats). MLflow = step 3 | §9.2 |
| S-2026-11-22-15 | **Segfault fixed before it shipped:** if faiss loads before torch, the next MiniLM encode segfaults on macOS (two OpenMP runtimes); my test module's top-level faiss import crashed the C5 MiniLM test. `dense.py` now imports faiss lazily, after torch. Checked both orders in a script | R2 only worked because it imported faiss after embedding |
| S-2026-11-22-16 | **Check run (not S1):** C3 × R-dense, top_k 20, dev 40 q: 0 unlocated spans, 19 views, 3 scan views correct; recall@budget 0.652 (n 33), mrr@10 0.551, entity recall 0.681; query embed p50 10 ms, search p50 0.04 ms; 100 s cold. Cold indexing C3 ≈ 0.07 s/page | Plumbing check only |
| S-2026-11-22-17 | **Open (Kousik, before S1):** `top_k` for S1. Config has 20 (API max, so the 1,200-word budget binds and small-chunk chunkers aren't capped); production default is 5 | Changes what recall@budget measures |

Changes: `evals/chunking/source.py` (`load_source(drop_pages=…)`), `evals/retrieval/{index_cache,views}.py`, `evals/retrievers/{__init__,dense}.py`, `evals/run.py`, `evals/configs/s1_chunkers.yaml`, `evals/retrieval/metrics.py` (`budget_pool`), `evals/tests/test_retrieval_runner.py` (13 tests). Evals 357 pass; app 173 pass, 2 skipped.

---

## 2026-11-22 (cont.) — R4 step 1: §8.3 retrieval metrics

**Branch:** `feature/r4-retrieval-metrics` (from `main` after PR #7). Kousik: "start step 1".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-7 | **R4 steps:** 1 metrics (pure, tested) → 2 runner + chunk/embedding cache → 3 local MLflow logging → 4 report generator → 5 run S1 on dev | Scope §12 R4; one module at a time | this log |
| S-2026-11-22-8 | **`evals/retrieval/metrics.py`:** all §8.3 QA metrics per question from a ranked chunk list. Coverage reuses R2's `covers()` / `locate_evidence()` (one rule for fragmentation and retrieval). Budget = copy of production `router._build_context` (greedy to 1,200 words, first chunk always in, first misfit stops the loop); a test pins `MAX_CONTEXT_WORDS` to production's | "after the same trimming production uses" |
| S-2026-11-22-9 | **Metric choices:** ndcg/map ideal = min(10, relevant chunks in the whole session index) (runner passes it; default = relevant in the ranking); context precision = RAGAS (mean precision@i at relevant ranks); precision@k divides by k even if fewer chunks come back; entity recall uses R3's `value_in` (so `62500.00` ≠ `$62,500`); noise ratio, cross-doc and amendment coverage are measured **within the budget**; `amend_only_superseded` = original span covered, no amending span; unanswerable → coverage metrics None, context_words/redundancy kept; unlocatable span = not covered, counted in `unlocated` | Definitions §8.3 leaves open |
| S-2026-11-22-10 | Corpus part per question: `simulated_scan` if any evidence on a scan, `consumer` for `cfpb*` sessions, else `edgar` (finalise strata) | §8.3 breakdown by part |

Changes: `evals/retrieval/{__init__,metrics}.py`, `evals/tests/test_retrieval_metrics.py` (14 tests). Evals 344 pass; app 173 pass, 2 skipped. Latency and recall-vs-exact metrics come with the runner (step 2) and S3.

---

## 2026-11-22 — R3: last 5 drafting jobs

**Branch:** `feature/r3-qa-golden`. Kousik: "run the remaining jobs".

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-22-1 | **Drafting complete: 92/92 jobs → 171 candidates, 13 rejected.** Last 5 (unanswerable-011…015): 8,349 tokens, $0.0018; 10/10 passed the filter, no draft issues. Unanswerable pool now 30 (quota 20). Measured over all 92 calls: mean 1,572 prompt + 508 completion. Existing review (cross_doc-003-a) not stale | Unanswerable now has slack |
| S-2026-11-22-2 | New unanswerable drafts repeat the 2 templates seen before (projected EBITDA/cash flow/NOI; internal rating/compliance officer/audit firm) — same pattern as S-2026-11-16-13. Not pre-reviewed by the agents | For Kousik's review: pick varied ones |
| S-2026-11-22-3 | **Review app bug fixed:** right panel (pages) empty and Keep/Drop/Later dead on every candidate with an agent suggestion. Cause: the suggestion panel reuses CSS class `.ev`; `markSpans()` looped over all `.ev` boxes, hit the panel (no evidence index) and threw, aborting `renderEditor` before the buttons were wired. Fix: select `#evs .ev` only. Checked in headless Chrome (PUT stubbed, no review written): pages + highlights render, Keep/Later send requests | Introduced in step 3b (`5dbd3fd`); Kousik reported it |
| S-2026-11-22-4 | **Review done (Kousik):** 171 candidates → 152 keep, 19 drop. First finalise run: cross_doc 13/15 (3 amendment candidates left unreviewed, probably missed during the app bug); Kousik reviewed them and kept them | — |
| S-2026-11-22-5 | **Finalise bug fixed:** the pick had no consumer cap, so the seeded fill took 44/120 consumer items (validator: EDGAR share ≥ 70% failed). The consumer-minimum step also took its 24 in type order (all lookup + table consumer), leaving calculation (11 EDGAR for 20 slots) and cross_doc to push consumer to 37. Fix in `pick()`: take each type's unavoidable consumer items first (they count toward the ≥ 20% minimum), then the minimum, then a fill capped at 36 consumer. 1 regression test (fails on the old code); evals 330 pass | Validator rule existed; the pick didn't respect it |
| S-2026-11-22-6 | **QA golden set written:** `evals/golden/qa/questions.jsonl`, q0001–q0120, 40 dev / 80 test (dev 8/8/7/5/5/7), types 25/25/20/15/15/20, EDGAR 84 / consumer 36, scan evidence 11, amendment 10 (exactly the minimum), 82 edited in review, all `llm_drafted_human_verified`. 32 kept questions unused (seed 20261116). `validate_set(final=True)` passes. From now on a fix = new qid + retire the old one (F3) | R3 deliverable |

---

## 2026-11-16 — R3 step 2d: drafting pilot (3 jobs)

**Branch:** `feature/r3-qa-golden`. Kousik committed step 2b–c (`1803284`) and OK'd the pilot.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-16-1 | **Pilot ran** `draft --run --limit 3` (all 3 = cross_doc jobs 001–003): 10,511 tokens, $0.0027. Measured per call: prompt 2,643 (token counter off by 3.6%), completion 861 of which reasoning 232; the guess was 1,400. Remaining 89 jobs: ≈ 125k prompt + ≈ 77–125k completion ≈ **200–250k tokens, 1.1–1.3 days, < $0.10** | Pilot replaces the completion guess |
| S-2026-11-16-2 | **Filter yield 4/6.** Both rejects (cross_doc-003) shortened the original-agreement span with "...", so it was not verbatim → correctly rejected. 2 kept candidates flagged "amendment without superseded_values" (deal_07: original was a $62,500 *closing* fee, amendment makes it an *annual facility* fee). Seen in review: entities sometimes hold a whole long definition (cross_doc-002) | Inputs for the prompt decision and Kousik's review |
| S-2026-11-16-3 | **Two prompt rules added** (Kousik): spans are never shortened with "..."/"…" (pick a shorter contiguous sentence instead); entities are short (a value or name, ≤ ~8 words), never a whole clause. Only cross_doc-003 re-drafted (old response kept in `data/results/r3/raw_superseded/`); 001–002 kept as drafted. Re-draft: 2/2 passed the filter; both ask about the old §2.7 terms, which the amendment removed ("[Reserved]"), so the right answer is "no longer applies" (for review) | Pilot lost 2/6 to "..." spans |
| S-2026-11-16-4 | **Full run stopped on a Groq 400 `json_validate_failed`** (cross_doc-011, LE↔CD): CFPB page text has raw `\x07` bullets; the model copied one into a span (correct per the rules) and a raw control character is invalid JSON. **Fix:** on that error code only, keep Groq's `failed_generation`, parse leniently (`strict=False`), save it as the raw response, record estimated tokens in the ledger (`usage_estimated`); any other API error still stops the run. 2 tests added (312 pass) | Spans must stay verbatim, so cleaning `\x07` out of the prompt would break the span check; salvage costs no extra tokens |
| S-2026-11-16-5 | **Run status end of day 1:** 37/92 jobs drafted → 66 candidates kept, 8 rejected; 87.2k tokens today, $0.024 total so far. Measured mean per call 1,698 prompt + 563 completion. Remaining 55 jobs ≈ 155k tokens (< 1 day). Second stop: Groq `json_validate_failed` with an **empty** generation on `table-007` (deal_07_ca p3 = a table-of-contents page on an image filing; OCR garbled the dot leaders so `pages.py`'s TOC filter missed it). Only page of its kind among the 92 jobs (≥ 10 SECTION/ARTICLE headings check) | Waiting on Kousik: how to handle table-007 |
| S-2026-11-16-6 | **QA review app** (Kousik: "build the review app") `evals/golden/qa/review_app/` (`server.py`, `store.py`, `index.html`; stdlib, 127.0.0.1:8766). Reads `drafts/candidates.jsonl`, writes decisions to **`drafts/reviews.json`** (separate file: candidates.jsonl is rebuilt by every draft run). Per candidate: keep / drop (reason required) / later, edited fields saved as `final`; **keep refused while F3 checks fail**; review marked **stale** if a re-draft changes the candidate (hash of draft fields). UI: shown pages with spans highlighted, prev/next page, page image (PyMuPDF, cached in `data/qa_review_images/`), select text → new evidence, live F3 check, kept-vs-quota bars. 9 tests (321 evals pass) | Same pattern as the ingestion labelling app; the finalise script reads `final` |
| S-2026-11-16-7 | **table-007 page swapped** (Kousik: option B) via `draft.PAGE_OVERRIDES`: deal_07_ca p3 (TOC) → **p101** (compliance-certificate schedule: Liquidity ≥ $20,000,000, add-back caps 15% / $240,000). Other unused deal_07_ca table pages rejected: p2/p4 are also TOC, p97–98 blank borrowing-base forms. Seeded selection untouched, so no drafted job moves. 1 test (322 pass) | Keeps table questions for the 25 quota |
| S-2026-11-16-8 | **Day 1 end:** run stopped at the daily budget (186.6k used today), **87/92 jobs drafted → 161 candidates kept, 13 rejected** (12 spans not verbatim, 1 page not shown). Kept per type: table 36, lookup 34, calculation 29, cross_doc 21, explain 21, unanswerable 20 (quota 25/25/20/15/15/20). One more `json_validate_failed` salvaged (table-016); table-007 on p101 drafted fine. Left: unanswerable-011…015 (≈ 14k tokens) → resume tomorrow. Cost so far ≈ $0.05 | Unanswerable has no slack until the last 5 jobs land |
| S-2026-11-16-9 | **Finalise script** `evals/golden/qa/finalise.py` (Kousik: "write the finalise script"): uses only non-stale "keep" reviews (`final` fields, verified_by/on from the review). **Pick** exact §5.1 type quotas, seed 20261116: constraint items first (amendment cross_doc ≥ 10, scan evidence ≥ 8, consumer ≥ 24 = 20%), then seeded fill; shortfall → nothing written. **Split** 40 dev / 80 test, dev per type by largest remainder (8/8/7/5/5/7), spread evenly over stratum order (consumer / scan / amendment / EDGAR). **qids** q0001… by type then candidate id; `source_cid`, `edited`, `drafted_by` kept. Writes only if `validate_set(final=True)` passes; no overwrite without `--force`. 5 tests (327 pass) | Deterministic and auditable; F3: fixes after release = new qid + retire |
| S-2026-11-16-10 | **Pool vs constraints (before review):** consumer 39 (need 24), scan 14 (need 8), amendment cross_doc 16 (need 10) | Review can drop at most 6 amendment, 6 scan, 15 consumer candidates before a constraint fails |
| S-2026-11-16-11 | **Agent pre-review, human decides** (Kousik chose this over agent-decides): two Claude agents check all 161 drafts against the page text (A: cross_doc/table/unanswerable, B: lookup/calculation/explain) and write suggestions only (`drafts/suggestions_a.json`, `_b.json`: keep / fix + exact edits / drop + reason, confidence). Review app shows a suggestion panel + filter; "Apply suggested edits" fills the form, Kousik still clicks Keep/Drop. Provenance stays `llm_drafted_human_verified`. 1 test (328 pass) | Speeds review without giving up human verification |
| S-2026-11-16-12 | **Pre-review result (161/161):** keep 39, fix 102, drop 20; all keep/fix pass F3 (re-checked by coordinator). If Kousik accepts all: lookup 34, table 32, unanswerable 20, calculation 20, explain 19, cross_doc 16; scan 12 (≥ 8), consumer 38 (≥ 24), amendment cross_doc 11 (≥ 10). **Tight:** calculation exactly 20 (8 rebuilt by the agent), amendment cross_doc 11. 18 fixes cite pages outside the drafting pages → app now also opens every page cited by the draft, review or suggestion (1 test, 329 pass) | Rebuilt questions need Kousik's check of the cited pages |
| S-2026-11-16-13 | **Drafting failure patterns found** (for the write-up and tomorrow's 5 jobs): fake calculations on pages without figures (15/29: arithmetic on section/page numbers, doc codes); amendment questions about values the amendment didn't change; -a/-b duplicates; same value differing across a session's documents (LE vs CD, 3 refinance CDs) → question must name the document; text-layer misread on CFPB H-25(C) ($96,500 is line L04, not Cash to Close); scan OCR shifting table rows (scan_38); unanswerable questions cloned from 2 templates (projected EBITDA, internal audit firm), some with false premises | Evidence for the "LLM drafts need human review" claim; candidate prompt rules |

---

## 2026-11-14 — R3 step 2c: drafting script (no real run yet)

**Branch:** `feature/r3-qa-golden`. Kousik: "go" on `draft.py`.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-14-1 | **Spend/token ledger** `evals/llm/ledger.py` → `data/spend_ledger.jsonl` (gitignored, F9 pattern): one line per call with prompt/completion/reasoning tokens and $ at list price; guards before each call: $5 per run, $40 total (override flag), **190k tokens per UTC day** (200k free tier minus margin) | D7; shared with S4 later |
| S-2026-11-14-2 | **Price gpt-oss-120b = $0.15 / $0.60 per M tokens** (in / out; reasoning billed as output), checked 2026-11-13 via aggregators quoting Groq's model docs (launch price was $0.75 out). qwen price left out until S4 needs it | D7 "published prices"; not verified on groq.com itself |
| S-2026-11-14-3 | **Drafting call:** gpt-oss-120b, reasoning effort low, temperature 0.4, JSON mode, max 6,000 completion tokens; one system prompt (span copied exactly, entities as written, answer_numeric plain number) + one rule per type (lookup, table, calculation, explain, unanswerable, cross_doc amendment, cross_doc LE↔CD); scan pages marked "OCR text"; calls paced under 8k tokens/min | Scope §5.2–5.3 fields; Groq free-tier per-minute limit |
| S-2026-11-14-4 | **Auto-reject only** (§5.3 step 2): span not found verbatim (exact or whitespace-insensitive) in the page it cites, a page the model wasn't shown, unanswerable with evidence, answerable without. **Everything else the F3 validator finds is kept as `draft_issues`** for Kousik's review | Scope says the script rejects on spans; Kousik decides the rest |
| S-2026-11-14-5 | **Resumable, reviewable output:** raw response per job in `data/results/r3/raw/` (never re-asked); candidates rebuilt from all raws every run → `evals/golden/qa/drafts/candidates.jsonl` (+ `rejected.jsonl` with reasons). Nothing writes `questions.jsonl` | A stopped run (budget, 429) loses nothing; review file is separate from the set |
| S-2026-11-14-6 | **Dry-run estimate (92 jobs, 184 candidates):** prompt 132,703 tokens (counted, o200k_harmony; max 3,138 per call) + completion guess 128,800 (300 per candidate + 800 reasoning per call) = **≈ 261,500 tokens ≈ 1.4 free-tier days, ≈ $0.10** at list price | Completion is a guess until a pilot measures it |

Changes: `evals/llm/ledger.py`, `evals/golden/qa/draft.py`, `evals/tests/test_qa_draft.py` (15 tests, fake client, no network). Evals 310 pass; production untouched.

### Open items
- Kousik: OK a **pilot of 3 jobs** (≈ 8.5k tokens), then re-estimate from measured tokens, then the full run (2 days).

---

## 2026-11-13 (cont.) — R3 step 2b: cross_doc pairs

**Branch:** `feature/r3-qa-golden`. Kousik: "go ahead" on `pairs.py`.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-24 | **Amendment pairing rule** (`evals/golden/qa/pairs.py`): an amendment page's targets = "Section N.NN … is (hereby) amended/restated/deleted/replaced" and 'definition of "Term" … amended', plus terms restated right after "Section 1.01/1.1 … is amended"; original page = first body page with a line starting "SECTION N.NN Title" or '"Term" (shall) mean(s)' | Scope §5.3: pair each amendment section with the original section it changes; deterministic, no LLM |
| S-2026-11-13-25 | **TOC pages skipped for pairing:** `pages.is_toc` or > 15 distinct section headings on the page (TOC pages ~75; body pages ≤ 9 — image filings double their text, so distinct). First version matched TOC pages in deal_03_ca / deal_07_ca (no page numbers there) | Pairs must point at the clause text |
| S-2026-11-13-26 | **Pair choice:** "money" pairs first (label names margin/rate/commitment/ratio/maturity/fee/covenant/…, word-bounded — "Integration" matched "rate" before), seeded round robin over deals, each amendment page and original page used once; 9 amendment jobs + 3 LE↔CD jobs (same form page, consumer session) | Money targets make "value after the amendment" questions with a superseded value (H9, stale-answer rate) |
| S-2026-11-13-27 | Bug fixed in the amendment regex: a sentence gap `[^.;]` stopped at the dot in "Section 1.01", missing 'definition of "X" in Section 1.01 … is amended' (5 pairs) | Caught by test |

Corpus: 47 pairs in 11 of 12 deals (deal_05: its amended sections aren't found in the original as headings), 19 money pairs. Chosen: 9 amendment jobs over 8 deals (fees, maturity dates ×4, commitments extension, rates, negative covenants, LIBOR successor) + 3 LE↔CD jobs. With `pages.py`: **92 jobs, no shortfall** (≈ 184 candidates). Note: 2 of the 9 (deal_03, deal_11) are image filings with OCR-damaged text (doubled letters lost).

Changes: `evals/golden/qa/pairs.py`, `evals/golden/qa/pages.py` (`Job.amendment`), `evals/tests/test_qa_pairs.py` (8 tests). Evals 295 pass; production untouched.

### Open items
- Next module: `draft.py` — prompts per type, Groq gpt-oss-120b call, token ledger, dry-run token estimate, span filter → `candidates.jsonl` for Kousik's review.

---

## 2026-11-13 (cont.) — R3 step 2a: drafting model + page selection

**Branch:** `feature/r3-qa-golden`. Kousik: start R3 step 2 (drafting script), one module at a time.

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-20 | **Drafting model = `openai/gpt-oss-120b`** (Kousik), `reasoning_effort=low`. Groq models on the account: qwen3.8-27b, gpt-oss-120b, gpt-oss-20b (+ non-text) | Drafting with the generator (qwen) would test it on questions shaped by its own wording; 120b is the strongest. Uses the judge's daily quota, which isn't needed before S4 | scope Decision Log |
| S-2026-11-13-21 | **≈ 1.5 candidates per final question** (Kousik): 80 single-page jobs + 12 cross_doc jobs, 2 candidates each ≈ 184 | Room for span rejects and Kousik's rejects without doubling review time | same |
| S-2026-11-13-22 | **Page selection** (`evals/golden/qa/pages.py`, seed 20261113): buckets edgar / consumer / scan; jobs per (type, bucket) in `PLAN` (consumer 19 jobs + scan 8 jobs, so the ≥ 20% consumer and ≥ 8 scan quotas have slack); type rules: table = table ≥ 3 rows on the page, calculation = table or ≥ 6 numbers, explain = EDGAR prose ≥ 200 words without a table, all ≥ 80 words; no CFPB covers, signature pages or TOC pages; sessions round robin, page at most once per type, unused pages first, a scan shares its use with its source page | Spread across deals; consumer forms have only 29 usable pages, so they're shared between types (4 pages) | same |
| S-2026-11-13-23 | Scan questions: evidence on the scan doc (its Apple Vision text); `session_docs` lists the whole session incl. the scan. Swapping the source page for its scan in the index is R4's job | The schema labels pages, not index contents | same |

Corpus check: 1,909 pages, usable edgar 1,384 / consumer 29 / scan 38 (TOC 32, signature 112, cover 11 excluded). All 80 single-page jobs filled, deterministic.

Changes: `evals/golden/qa/pages.py`, `evals/tests/test_qa_pages.py` (9 tests). Evals 287 pass; production untouched.

### Open items
- Next module: `pairs.py` (amendment section ↔ original section, LE ↔ CD pairs → 12 cross_doc jobs). Then `draft.py` (Groq + token estimate).

---

## 2026-11-13 (cont.) — R3 step 1: QA golden-set schema + F3 validator

**Branch:** `feature/r3-qa-golden`, built on `feature/r2-chunkers` (Kousik: "start R3 now on a branch built on R2"). R2 PR #6 opened (depends on #5; merge #5 first).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-15 | **`amendment: bool` added to the §5.2 schema** | H9 and the "≥ 10 amendment questions" quota need to know which questions are amendment questions; inferring it from document roles is ambiguous (a lookup on an amendment isn't an amendment question). Amendment ⇒ `superseded_values` non-empty and evidence from an amendment doc; otherwise `superseded_values` empty | scope Decision Log |
| S-2026-11-13-16 | Sessions come from the manifest `deal_id` (EDGAR deals, CFPB scenarios), scans included; `session_docs` must all belong to `session_id` | One source of truth for sessions | same |
| S-2026-11-13-17 | **Validator text = parse `downstream_config("P2", part)` + cleaner v2**, the text chunkers index (scope: "parser config chosen in S0"). Span match: exact, else whitespace-insensitive | Same text retrieval scores against | same |
| S-2026-11-13-18 | **Entity / superseded-value match:** numeric values match on `numeric_tokens` (all present, with multiplicity), others on `norm_match` substring | Scope §8.3 "numbers, rates and dates after normalization; names by normalized string"; reuses the S0 normalizer | same |
| S-2026-11-13-19 | Validator also enforces: nothing without `verified_by`/`verified_on`; unanswerable ⇔ type `unanswerable` ⇔ no evidence; cross_doc evidence from 2+ docs; calculation needs `answer_numeric`; tolerance needs a number. `--final` adds the §5.1 quotas (type counts, 40/80 split, ≥ 70% EDGAR, ≥ 20% consumer, ≥ 8 scan-evidence, ≥ 10 amendment cross_doc) | §5.1–5.3, F3 | same |

Changes: `evals/golden/qa/schema.py`, `evals/golden/qa/validate.py` (CLI: `venv/bin/python -m evals.golden.qa.validate [--final]`), `evals/tests/test_qa_schema.py` (17 tests). Smoke test on real text: a correct CFPB span ("Loan Amount $162,000 NO", cfpb_cd_h25b p2) passes, an altered one is caught. Evals 278 pass.

### Open items
- Next: R3 step 2, the drafting script (Groq; estimate tokens before running) — ask Kousik first.

---

## 2026-11-13 (cont.) — R2: open chunker questions settled

**Branch:** `feature/r2-chunkers`. Kousik answered the three open questions (each with the agent's recommendation).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-10 | **C4 promotes run-in headings** (Kousik): a `text`/`list` element opening with a dotted section number + a title of ≤ 15 title-case words ending in "." ("Section 2.11 Prepayments. (a) …", "8.3.4.3 Financial Statements. …", "SECTION 2.05 — Funding of Borrowings. …", "SECTION 2.03. [Reserved].") is split into a heading and its body (`c4_layout.split_runin`). Rejected: TOC lines (rest is a page number / dot leaders), lowercase titles (garbled text, amendment instructions "1.1. Clause (a) of Section 5.08 … is amended", numbered sub-clauses), titles without a closing period. `runin_headings=False` = strict C4. **Deviation from §6.1** "boundaries from Docling elements" | Without it, 56% of numbered sections are invisible to C4 (S-2026-11-12-28), so H1 would test Docling's labels more than layout-aware chunking | scope Decision Log |
| S-2026-11-13-11 | **C2 keeps whole-sentence overlap** (Kousik). 46% of boundaries have none (S-2026-11-12-23); reported with its results | C2's point is sentence alignment; a token fallback would make it C3-like | same |
| S-2026-11-13-12 | **C5 stays literal p95** (Kousik). 89% of cuts are size-driven (S-2026-11-13-3), each still at the segment's largest distance; reported with its results | Any other percentile would be arbitrary without QA data; S1 judges C5 | same |
| S-2026-11-13-13 | Rule check on all 1,023 EDGAR body elements starting "N.N…": 945 promoted. Missed on purpose/accepted: OCR-split titles ("Use of Proce ds."), titles without a period ("Section 5.10 Margin Stock") | Sampled and read before fixing the rule | — |
| S-2026-11-13-14 | `run_metrics --chunkers …` now merges into existing results (same table config + cleaning version) instead of overwriting | Re-running one chunker shouldn't re-embed all five | `run_metrics.save` |

Effect on the corpus (C4, P2/P2V, cleaner v2): run-in headings promoted EDGAR 945, scans 8, consumer 0. EDGAR distinct sections 1,215 → 1,971; C4 chunks 8,503 → **8,749**. Corpus check: 0 errors, uncovered 0.28% (unchanged: page furniture only). §8.2 for C4 now: MiniLM p5/p50/p95/max 70/208/254/254, truncation 0%, table-break 0% (34.7% split, header always repeated), sentence-break 8.0%, redundancy 5.0%, index 12.8 MB. Full tables regenerated: `docs/21_r2_chunk_metrics.md`.

Changes: `evals/chunking/c4_layout.py` (`split_runin`, `runin_headings` param), `run_metrics.py` (`save` merge), tests (+18: run-in split/reject cases, C4 integration incl. strict mode with identical coverage, partial-run merge). Evals 261 pass; production untouched.

### Open items
- None left for R2. Next: open the R2 PR `feature/r2-chunkers` → `main`, then R3 (QA golden set) — ask Kousik.

---

## 2026-11-13 (cont.) — R2 step 5: §8.2 chunk metrics

**Branch:** `feature/r2-chunkers`. Kousik: "R2 step 5 go" (C2 overlap, C4 run-in headings, C5 tuning still open; all chunkers left as built).

| ID | Decision / finding | Why | Recorded in |
|---|---|---|---|
| S-2026-11-13-4 | §8.2 metrics computed from chunk **spans** (one definition for all chunkers); text a chunker adds (C1 headings/pipes, C4 prefixes) counts in size and truncation only | Provenance is the common ground; see `evals/chunking/metrics.py` docstring | scope Decision Log |
| S-2026-11-13-5 | Definitions: truncation = > 254 MiniLM content tokens, per chunk copy + page words no chunk embeds; table-break = split and some piece touching a data row lacks the header row; sentence-break = chunk start/end not a syntok sentence edge (doc start/end excluded); redundancy = duplicate MiniLM page-token copies ÷ all copies; fragmentation = §8.3 cover rule (≥ 50% of non-blank chars) | Scope §8.2 wording made operational | same |
| S-2026-11-13-6 | Sentence reference = C2's splitter (syntok per line), so C2/C5 only break where they split an over-long sentence: favoured by construction (disclosed) | No independent sentence labels | same |
| S-2026-11-13-7 | Index = one FAISS IndexFlatIP per chunker over the corpus; memory = n × 384 × 4 B, disk = written file; embedding = production MiniLM, normalized, batch 64, CPU | Flat size is linear, so total = sum over sessions | same |
| S-2026-11-13-8 | **H8 measured: supported.** C1 truncation 99.0% [98.1, 99.4] (> 50%); C2–C5 0.0% (< 5%). 23.0% [22.5, 23.4] of corpus words are embedded by no C1 chunk (R0's loan1: 22.5%) | Formal hypothesis reporting stays in R8 | scope §11 note |
| S-2026-11-13-9 | Evidence fragmentation: implemented and tested (`metrics.fragmentation`), **pending the QA set (R3)** | Needs evidence spans | — |

Changes: `evals/chunking/metrics.py`, `evals/chunking/run_metrics.py`, `evals/chunking/r2_report.py` (renders `docs/21_r2_chunk_metrics.md`, F7), `evals/tests/test_chunking_metrics.py` (23 tests); `docs/19_handoff.md` brought up to date. Evals 244 pass; production untouched.
Bugs found by the corpus run and fixed before the final run: C1 word alignment failed across a markdown `| --- |` separator row (9,227 → 2,178 unaligned words); BOM-only "words" counted as never embedded (0.01% for C2/C3/C5 → 0%).

Corpus run (92 docs, P2/P2V, cleaner v2, 0 missing; output `data/results/r2/chunk_metrics.json` + per-doc `chunk_metrics_docs.jsonl`, gitignored):

| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Chunks | 4,774 | 9,760 | 6,583 | 8,503 | 10,744 |
| MiniLM tokens p5/p50/p95/max | 380/476/711/1,954 | 77/176/200/200 | 253/254/254/254 | 74/213/254/254 | 54/139/200/200 |
| tiktoken p5/p50/p95/max | 381/483/699/1,020 | 75/175/210/288 | 238/253/279/378 | 77/215/263/338 | 53/141/203/288 |
| Words p5/p50/p95/max | 301/358/531/600 | 55/129/162/190 | 159/194/214/242 | 50/157/200/237 | 38/104/160/186 |
| ★ Truncation rate [95% CI] | **99.0%** [98.1, 99.4] | 0% | 0% | 0% | 0% |
| Tokens dropped per chunk copy | 49.9% | 0 | 0 | 0 | 0 |
| ★ Page words never embedded [CI] | **23.0%** [22.5, 23.4] | 0% | 0% | 0.26% [0.20, 0.38] | 0% |
| Tables split / broken (n = 216) | 13.0% / 13.0% | 49.1% / 49.1% | 46.8% / 46.8% | 34.7% / **0%** | 56.9% / 56.9% |
| Sentence-break rate | 6.9% | 12.3% | 97.6% | 7.9% | 10.4% |
| Overlap redundancy | 38.8% | 8.4% | 12.6% | 5.4% | 0% |
| Index memory = disk | 7.0 MB | 14.3 MB | 9.6 MB | 12.5 MB | 15.7 MB |
| Chunking / embedding time (s) | 0.5 / 83 | 9.6 / 127 | 4.8 / 102 | 8.2 / 128 | 163 / 115 |

Notes:
- C1: 2,178 page words in truncated chunks couldn't be aligned to the chunk text and count as not embedded; counting them all as embedded gives 22.76%, so the bound is tight. C1 truncation by part: EDGAR 99.6%, consumer 92.2%, scans 78.4%.
- C1's 38.8% redundancy includes its tables, which appear twice (page text + markdown) when the two copies land in different chunks.
- C4's never-embedded 0.26% = page headers/footers/numbers it leaves out on purpose (consumer 3.4%, scans 1.1%, EDGAR 0.2%).
- C1's low table-break (13%) is because it also writes each table whole as markdown; its page-text copy of the rows is split more often.
- C5's chunking time includes embedding every sentence. Timings are from one laptop run, so they're rough.

### Open items
- Still open: C2 overlap (step 2), C4 run-in headings (step 3), C5 tuning (step 4).
- R2 deliverables now complete pending those answers → next: PR `feature/r2-chunkers` → main, then R3 (QA golden set). Ask Kousik first.

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
