"""
Results report for the retrieval stages (scope v2 §9.2 report.py, §9.6 statistics). Numbers are
generated from the run folders, never typed by hand (F7).

    venv/bin/python -m evals.report --stage S1 --split dev        # -> docs/22_s1_retrieval.md
    venv/bin/python -m evals.report --stage S2 --split dev        # -> docs/23_s2_retrieval.md
    venv/bin/python -m evals.report --stage S3 --split dev        # -> docs/24_s3_index_types.md (no pick)

Reads data/results/r4/<stage>/<split>/<run>/{summary.json, questions.jsonl} for every run in the
stage config, in config order. All runs must have scored the same questions file (sha256) and the
same question ids, manifest, embedder, parse config and cleaner; otherwise the report refuses (no
mixed comparisons). A stage config may name an earlier stage's run as `baseline` (S2: S1's C1-dense);
it is loaded first, under the same checks, and is not a pick candidate.

Stage-specific sections: S2 judges H2 and H3 (§11) with the rules in `hypotheses_s2`, and shows
latency per component and rerank truncation.

Statistics (§9.6):
- ★ metrics (recall@budget, entity recall): mean with a question-bootstrap 95% CI
  (evals.stats.bootstrap_ci, 10,000 resamples, seed 0).
- Paired comparisons on ★ metrics: per-question differences, bootstrap CI and p (paired_diff_ci).
  "Better" only when the CI excludes 0. Every run vs the baseline (first run in the config: C1 +
  R-dense = production), plus neighbours in the ranking (is #2 distinguishable from #3?). Holm-
  adjusted p over the baseline comparisons shown for information; §9.6 requires Holm only in S4.
- The pick (S1: top 2 chunkers) is by recall@budget point estimate (D3: one decision metric).
- Every other metric: point estimate, labelled diagnostic.
- Evidence fragmentation (★ in §8.2): spans no single chunk in the question's index covers ÷
  located spans, pooled, with a question-bootstrap CI (ratio_ci).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from evals.run import OUT_ROOT, REPO, load_config
from evals.stats import bootstrap_ci, holm, paired_diff_ci, ratio_ci

STARRED = ("recall@budget", "entity_recall")
PICKS = {"S1": 2, "S2": 3}                     # §9.3: S1 top 2 chunkers, S2 top 3 configurations
DOCS = {("S1", "dev"): REPO / "docs" / "22_s1_retrieval.md", ("S2", "dev"): REPO / "docs" / "23_s2_retrieval.md",
        ("S3", "dev"): REPO / "docs" / "24_s3_index_types.md"}
CONFIGS = {"S1": "s1_chunkers.yaml", "S2": "s2_retrievers.yaml", "S3": "s3_index_types.yaml"}
SAME = ("questions_sha256", "manifest_sha256", "embedder", "table_config", "cleaning_version")
TYPES = ("lookup", "table", "calculation", "cross_doc", "explain", "unanswerable")
PARTS = ("edgar", "consumer", "simulated_scan")
DIAGNOSTIC = ("recall@1", "recall@3", "recall@5", "recall@10", "hit@1", "hit@5", "hit@10",
              "precision@1", "precision@5", "precision@10", "context_precision", "mrr@10", "ndcg@10",
              "map@10", "noise_ratio", "redundancy", "context_words", "cross_doc_coverage",
              "amend_coverage", "amend_only_superseded")


# ---- loading -----------------------------------------------------------------------------------
def load_runs(stage: str, split: str, names: list[str], root: Path = OUT_ROOT,
              baseline: dict | None = None) -> dict[str, dict]:
    """Runs in config order; an external baseline ({stage, run}) first, flagged `external`."""
    runs: dict[str, dict] = {}
    todo = ([(baseline["stage"], baseline["run"], True)] if baseline else []) + [(stage, n, False) for n in names]
    for st, name, external in todo:
        d = root / st.lower() / split / name
        if not (d / "summary.json").exists():
            raise FileNotFoundError(f"run {name}: no results in {d} (run evals.run first)")
        summary = json.loads((d / "summary.json").read_text())
        rows = [json.loads(line) for line in (d / "questions.jsonl").open()]
        runs[name] = {"summary": summary, "rows": {r["qid"]: r for r in rows}, "external": external,
                      "stage": st}
    qids = {tuple(sorted(r["rows"])) for r in runs.values()}
    if len(qids) > 1 or any(len({json.dumps(r["summary"].get(k)) for r in runs.values()}) > 1 for k in SAME):
        raise ValueError("runs scored different question sets or setups; re-run the stage before reporting")
    if any(r["summary"].get("limit") is not None for r in runs.values()):
        raise ValueError("a run used --limit (smoke run); re-run without --limit before reporting")
    return runs


def values(run: dict, metric: str, qids: list[str]) -> list[float | None]:
    return [run["rows"][q]["metrics"].get(metric) for q in qids]


# ---- formatting --------------------------------------------------------------------------------
def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def p_cell(p) -> str:
    """Bootstrap p; 0 means no resample crossed 0, i.e. p < 1 / 10,000."""
    return "–" if p is None else ("< 0.0001" if p == 0 else f"{p:.4f}")


def ci_cell(c: dict) -> str:
    return "–" if c["point"] is None else f"**{c['point']:.3f}** [{c['lo']:.3f}, {c['hi']:.3f}]"


def diff_cell(d: dict) -> str:
    if d["diff"] is None:
        return "–"
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def verdict(d: dict, a: str, b: str) -> str:
    if d["better"] == "b":
        return f"{b} better"
    if d["better"] == "a":
        return f"{a} better"
    return f"no detectable difference (n = {d['n']})"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def retriever_label(rc: dict) -> str:
    out = rc["retriever"] + ("+rerank" if rc.get("rerank") else "")
    return out + (f" (parents {rc['parent_chunker']})" if rc.get("parent_chunker") else "")


# ---- S2 hypotheses (§11), rules fixed before the S2 run (S-2026-11-23-…) -------------------------
_QUOTED = re.compile(r'["“”]|(?:^|\s)[\'‘][^\'’]+[\'’]')
_SECTION = re.compile(r"\b(?:Section|Sections|clause|Article|§)\s*[\w(]|\([a-z]{1,4}\)|\(\d{1,2}\)")
_TITLE_RUN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|for|to)?\s*[A-Z][a-zA-Z]+)+")


def has_identifier(question: str) -> bool:
    """H2's "lookup questions that contain exact identifiers (rates, amounts, section numbers, defined
    terms)", as a fixed rule: the question has a digit (rates, amounts, dates, years), a quoted term, a
    section/clause reference, or a defined term = a run of >= 2 capitalized words that doesn't start
    the question ("Administrative Agent", "Annual Percentage Rate")."""
    if re.search(r"\d", question) or _QUOTED.search(question) or _SECTION.search(question):
        return True
    return any(m.start() > 0 for m in _TITLE_RUN.finditer(question))


def _find(runs: dict, **want) -> str | None:
    for name, run in runs.items():
        rc = run["summary"]["run"]
        if not run.get("external") and all((rc.get(k) or None) == (v or None) for k, v in want.items()):
            return name
    return None


def _overall(verdicts: list[str]) -> str:
    """supported: every comparison's CI above 0; not supported: any CI below 0; else inconclusive."""
    if not verdicts:
        return "not tested (runs missing)"
    if "below" in verdicts:
        return "**not supported**"
    return "**supported**" if all(v == "above" for v in verdicts) else "**inconclusive**"


def _side(d: dict) -> str:
    return "above" if d["better"] == "b" else "below" if d["better"] == "a" else "includes 0"


def hypotheses_s2(runs: dict, qids: list[str], questions: dict[str, dict]) -> str:
    out = []
    # H2: R-hybrid beats R-dense on lookup questions with exact identifiers (recall@budget; mrr@10 shown)
    sub = [q for q in qids if questions[q]["type"] == "lookup" and has_identifier(questions[q]["question"])]
    n_lookup = sum(questions[q]["type"] == "lookup" for q in qids)
    rows, verdicts = [], []
    for ch in sorted({r["summary"]["run"]["chunker"] for r in runs.values() if not r.get("external")}):
        dense, hyb = _find(runs, chunker=ch, retriever="R-dense", rerank=False), \
            _find(runs, chunker=ch, retriever="R-hybrid", rerank=False)
        if not (dense and hyb):
            continue
        d = paired_diff_ci(values(runs[dense], "recall@budget", sub), values(runs[hyb], "recall@budget", sub))
        m = paired_diff_ci(values(runs[dense], "mrr@10", sub), values(runs[hyb], "mrr@10", sub))
        verdicts.append(_side(d))
        rows.append([f"`{hyb}` − `{dense}`", str(d["n"]), diff_cell(d), _side(d), diff_cell(m)])
    out.append(f"**H2** — `R-hybrid` beats `R-dense` on `lookup` questions that contain exact identifiers. "
               f"Subset: {len(sub)} of {n_lookup} dev lookup questions match the identifier rule "
               f"(`report.has_identifier`: digit, quoted term, section/clause reference, or a defined term = "
               f"≥ 2 capitalized words not at the start); unanswerable ones drop out of the metric. Test: paired "
               f"recall@budget difference per chunker, no rerank. Rule: supported if every CI is above 0, not "
               f"supported if any is below 0, else inconclusive.\n\n" +
               md_table(["Comparison", "n", "recall@budget diff [95% CI]", "CI vs 0", "mrr@10 diff [95% CI] (diagnostic)"],
                        rows) + f"\n\nVerdict (dev): {_overall(verdicts)}.")
    # H3: reranking improves mrr@10 for every base retriever
    rows, verdicts = [], []
    for name, run in runs.items():
        rc = run["summary"]["run"]
        if run.get("external") or not rc.get("rerank"):
            continue
        base = _find(runs, chunker=rc["chunker"], retriever=rc["retriever"],
                     parent_chunker=rc.get("parent_chunker"), rerank=False)
        if base is None:
            continue
        d = paired_diff_ci(values(runs[base], "mrr@10", qids), values(run, "mrr@10", qids))
        verdicts.append(_side(d))
        rows.append([f"`{name}` − `{base}`", str(d["n"]), diff_cell(d), _side(d)])
    out.append("**H3** — Reranking improves `mrr@10` for every base retriever. Test: paired mrr@10 "
               "difference, reranked − base, every base configuration in S2. Same rule.\n\n" +
               md_table(["Comparison", "n", "mrr@10 diff [95% CI]", "CI vs 0"], rows) +
               f"\n\nVerdict (dev): {_overall(verdicts)}.")
    return "\n\n".join(out)


def latency_table(runs: dict) -> str:
    comps = []
    for run in runs.values():
        for c in run["summary"]["latency_ms"]:
            if c not in comps and c != "total":
                comps.append(c)
    comps.append("total")
    rows = []
    for name, run in runs.items():
        lat = run["summary"]["latency_ms"]
        rows.append([f"`{name}`"] + [("–" if c not in lat or lat[c]["p50"] is None
                                      else f"{lat[c]['p50']:.1f} / {lat[c]['p95']:.1f}") for c in comps])
    return md_table(["Run"] + [f"{c} p50 / p95" for c in comps], rows)


def rerank_table(runs: dict) -> str:
    rows = []
    for name, run in runs.items():
        info = run["summary"].get("retriever_info") or {}
        if info.get("rerank_candidates"):
            n, t = info["rerank_candidates"], info["rerank_truncated"]
            rows.append([f"`{name}`", str(n), str(t), f"{t / n:.1%}"])
    if not rows:
        return ""
    return ("Candidates whose (question, chunk) pair exceeds the cross-encoder's 512 tokens are cut from the "
            "chunk's end before scoring.\n\n" +
            md_table(["Run", "candidates scored", "truncated", "share"], rows))


# ---- sections ----------------------------------------------------------------------------------
def headline(runs: dict, qids: list[str]) -> tuple[str, dict]:
    stats: dict[str, dict] = {}
    rows = []
    for name, run in runs.items():
        s = {m: bootstrap_ci(values(run, m, qids)) for m in STARRED}
        frag_rows = [run["rows"][q] for q in qids if run["rows"][q].get("fragmented") is not None]
        located = [r["spans"] - r["unlocated"] for r in frag_rows]
        s["fragmentation"] = ratio_ci([r["fragmented"] for r in frag_rows], located)
        stats[name] = s
        o = run["summary"]["breakdown"]["overall"]
        lat = run["summary"]["latency_ms"]["total"]
        rc = run["summary"]["run"]
        rows.append([name + (f" ({run['stage']})" if run.get("external") else ""), rc["chunker"],
                     retriever_label(rc), str(rc["top_k"]), ci_cell(s["recall@budget"]),
                     ci_cell(s["entity_recall"]), f3(o["mrr@10"]["mean"]), f3(o["hit@10"]["mean"]),
                     f"{o['context_words']['mean']:.0f}", ci_cell(s["fragmentation"]),
                     f"{lat['p50']:.1f} / {lat['p95']:.1f}"])
    n = stats[next(iter(stats))]["recall@budget"]["n"]
    table = md_table(["Run", "Chunker", "Retriever", "top_k", f"★ recall@budget (n={n})", "★ entity recall",
                      "mrr@10", "hit@10", "context words", "★ evidence fragmentation", "retrieval ms p50 / p95"],
                     rows)
    return table, stats


def rank_order(runs: dict, stats: dict) -> list[str]:
    """By recall@budget (D3); ties -> faster (lower retrieval latency p50), as S0's "ties -> faster"
    (Kousik 2026-11-22, S-2026-11-22-30)."""
    return sorted([r for r in runs if not runs[r].get("external")],
                  key=lambda r: (-(stats[r]["recall@budget"]["point"] or 0),
                                       runs[r]["summary"]["latency_ms"]["total"]["p50"]))


def ranking(runs: dict, stats: dict, stage: str) -> tuple[str, list[str]]:
    order = rank_order(runs, stats)
    k = PICKS.get(stage, 2)
    lines = [f"{i}. `{r}` — recall@budget {stats[r]['recall@budget']['point']:.3f}" +
             (" ← **picked**" if i <= k else "") for i, r in enumerate(order, 1)]
    if len(order) > k:
        cut = stats[order[k - 1]]["recall@budget"]["point"]
        tied = [r for r in order if stats[r]["recall@budget"]["point"] == cut]
        if len(tied) > 1 and any(order.index(r) >= k for r in tied):
            lat = ", ".join(f"`{r}` {runs[r]['summary']['latency_ms']['total']['p50']:.1f} ms" for r in tied)
            lines.append(f"\n**Tie at the pick boundary:** {', '.join(f'`{r}`' for r in tied)} have the same "
                         f"recall@budget. Tie rule: faster wins (retrieval latency p50: {lat}), as in S0.")
    return "\n".join(lines), order[:k]


def comparisons(runs: dict, qids: list[str], order: list[str]) -> str:
    names = list(runs)
    base = names[0]
    parts = []
    for metric in STARRED:
        rows, pvals = [], []
        for other in names[1:]:
            d = paired_diff_ci(values(runs[base], metric, qids), values(runs[other], metric, qids))
            pvals.append(d["p"] if d["p"] is not None else 1.0)
            rows.append([f"`{other}` − `{base}`", diff_cell(d), p_cell(d["p"]), None, verdict(d, base, other)])
        for row, p in zip(rows, holm(pvals) if pvals else []):
            row[3] = p_cell(p)
        for a, b in zip(order, order[1:]):
            d = paired_diff_ci(values(runs[b], metric, qids), values(runs[a], metric, qids))
            rows.append([f"`{a}` − `{b}` (ranks {order.index(a) + 1}–{order.index(b) + 1})", diff_cell(d),
                         p_cell(d["p"]), "", verdict(d, b, a)])
        parts.append(f"**{metric}** (paired, per-question differences, 95% bootstrap CI)\n\n" +
                     md_table(["Comparison", "Mean difference [95% CI]", "p", "Holm p (vs baseline)", "Verdict"],
                              rows))
    return "\n\n".join(parts)


def breakdown_table(runs: dict, metric: str, groups: list[tuple[str, str]]) -> str:
    rows = []
    for label, key in groups:
        cells, n = [], None
        for run in runs.values():
            g = run["summary"]["breakdown"].get(key, {}).get(metric)
            cells.append(f3(g["mean"]) if g else "–")
            n = g["n"] if g else n
        if n is not None:
            rows.append([label, str(n)] + cells)
    return md_table(["Group", "n"] + [f"`{r}`" for r in runs], rows)


def diagnostics(runs: dict) -> str:
    rows = []
    for m in DIAGNOSTIC:
        cells = []
        for run in runs.values():
            g = run["summary"]["breakdown"]["overall"].get(m)
            cells.append("–" if not g else (f"{g['mean']:.0f}" if m == "context_words" else f3(g["mean"])))
        rows.append([m] + cells)
    return md_table(["Metric (diagnostic)"] + [f"`{r}`" for r in runs], rows)


def provenance(runs: dict) -> str:
    s0 = next(iter(runs.values()))["summary"]
    rows = [[f"`{n}`", r["summary"]["git_sha"][:12], str(r["summary"]["questions"]),
             str(r["summary"]["cache"]), str(r["summary"].get("mlflow_run_id", "–"))[:12],
             str(r["summary"]["seconds"])] for n, r in runs.items()]
    head = (f"Questions file sha256 `{s0['questions_sha256'][:16]}…`, manifest `{s0['manifest_sha256'][:16]}…`, "
            f"embedder `{s0['embedder']}`, parse `{s0['table_config']}` (scans `P2V`), cleaner "
            f"`{s0['cleaning_version']}`.")
    return head + "\n\n" + md_table(["Run", "git SHA", "questions", "cache", "MLflow run", "seconds"], rows)


def notes(runs: dict, qids: list[str]) -> str:
    """Caveats with their numbers computed from the runs (F7)."""
    first = next(iter(runs.values()))
    over = first["summary"]["breakdown"]["overall"]
    n_ans = over["recall@budget"]["n"]
    n_unans = len(qids) - n_ans
    n_cross = over.get("cross_doc_coverage", {}).get("n", 0)
    n_amend = over.get("amend_coverage", {}).get("n", 0)
    frag = sum(r.get("fragmented") or 0 for run in runs.values() for r in run["rows"].values())
    out = [f"- n = {n_ans} answerable dev questions ({n_unans} unanswerable left out of coverage metrics). "
           "Type and part breakdowns have single-digit n: read them as diagnostics, not results.",
           f"- Cross-document coverage (n = {n_cross}) and amendment coverage (n = {n_amend}) are too small "
           "to compare chunkers; H9 is judged on the full set in R8.",
           "- The test split is untouched: S1–S3 pick on dev only (§9.3)."]
    if any(r["summary"]["run"]["retriever"] == "R-pc" for r in runs.values()):
        out.append("- `R-pc` returns parents built per question, so its ndcg@10 / map@10 use the children's "
                   "relevant count as the ideal: diagnostic only, not comparable with other retrievers. "
                   "recall@budget, entity recall and mrr@10 don't depend on it (S-2026-11-22-44).")
    ext = [n for n, r in runs.items() if r.get("external")]
    if ext:
        out.append(f"- Baseline `{ext[0]}` is the {runs[ext[0]]['stage']} run (same questions file, manifest, "
                   "embedder, parse config and cleaner — checked); it is compared against, not picked.")
    if frag == 0:
        out.append("- **Evidence fragmentation is 0 for every chunker.** QA evidence spans are short (about one "
                   "sentence), so each span lies at least 50% inside a single chunk of every chunker. On this "
                   "question set the metric does not separate chunkers.")
    return "\n".join(out)


def render(stage: str, split: str, runs: dict, questions: dict[str, dict] | None = None) -> str:
    qids = sorted(next(iter(runs.values()))["rows"])
    head_table, stats = headline(runs, qids)
    rank_text, _ = ranking(runs, stats, stage)
    order = rank_order(runs, stats)
    base = next(iter(runs))
    types = [(f"type = {t}", f"type={t}") for t in TYPES]
    parts = [(f"part = {p}", f"part={p}") for p in PARTS]
    extra = ""
    if stage == "S2":
        extra = f"""## Hypotheses (§11, judged on dev)

{hypotheses_s2(runs, qids, questions or {})}

## Retrieval latency by component (ms)

Per question, after the session index is built (index building is offline, as in production). Rerank runs
on CPU.

{latency_table(runs)}

## Rerank truncation

{rerank_table(runs)}

"""
    return f"""# {stage} Results — Retrieval ({split} split)

**Generated** by `venv/bin/python -m evals.report --stage {stage} --split {split}` from
`data/results/r4/{stage.lower()}/{split}/` (runs: `venv/bin/python -m evals.run --config …`). Do not edit
numbers by hand (F7). Definitions: `evals/retrieval/metrics.py`; statistics: `evals/report.py` docstring;
decisions: `docs/18_session_log.md` (S-2026-11-22-*).

★ = headline metric (95% bootstrap CI over questions). Everything else is diagnostic (point estimate).
Unanswerable questions are left out of coverage metrics. Baseline = `{base}` (production configuration).

## Headline

{head_table}

## Pick (decision metric: recall@budget)

{rank_text}

The pick follows the point estimate (D3). Whether neighbours in the ranking are distinguishable is in the
next table: if the CI includes 0, the pick between them is not supported by the data at this n.

## Paired comparisons

{comparisons(runs, qids, order)}

## recall@budget by question type

{breakdown_table(runs, "recall@budget", types)}

## recall@budget by corpus part

{breakdown_table(runs, "recall@budget", parts)}

## Entity recall by question type

{breakdown_table(runs, "entity_recall", types)}

## Diagnostic metrics (overall)

{diagnostics(runs)}

{extra}## Notes

{notes(runs, qids)}

## Provenance

{provenance(runs)}
"""


# ---- S3: index types (no pick) -----------------------------------------------------------------
def index_label(spec: dict | None) -> str:
    from evals.retrievers.ann import parse_spec
    sp = parse_spec(spec)
    if sp["type"] == "hnsw":
        return f"HNSW M={sp['M']} efC={sp['ef_construction']} efSearch={sp['ef_search']}"
    if sp["type"] == "ivf":
        return f"IVF nlist={sp['nlist']} nprobe={sp['nprobe']}"
    return "Flat (exact)"


def render_s3(split: str, runs: dict, pooled: dict) -> str:
    qids = sorted(next(iter(runs.values()))["rows"])
    names = list(runs)
    base = names[0]
    rows = []
    for name, run in runs.items():
        sm = run["summary"]
        ci = bootstrap_ci(values(run, "recall@budget", qids))
        d = paired_diff_ci(values(runs[base], "recall@budget", qids), values(run, "recall@budget", qids))
        info = sm.get("retriever_info") or {}
        ov = info["exact_overlap"] / info["exact_overlap_n"] if info.get("exact_overlap_n") else None
        vs, tot = sm["latency_ms"]["vector_search"], sm["latency_ms"]["total"]
        rr = sm["latency_ms"].get("rerank", {}).get("p50")
        rows.append([f"`{name}`", index_label(sm["run"].get("index")), ci_cell(ci),
                     "–" if name == base else diff_cell(d), "1 (exact)" if name == base else f3(ov),
                     f"{1000 * vs['p50']:.0f} / {1000 * vs['p95']:.0f}", "–" if rr is None else f"{rr:.0f}",
                     f"{tot['p50']:.0f}"])
    n = bootstrap_ci(values(runs[base], "recall@budget", qids))["n"]
    session = md_table(["Run", "Index", f"★ recall@budget (n={n})", "diff vs Flat [95% CI]",
                        "dense top-20 recall vs exact (mean)", "vector search µs p50 / p95", "rerank ms p50",
                        "total ms p50"], rows)
    rr = [r["summary"]["latency_ms"]["rerank"]["p50"] for r in runs.values() if "rerank" in r["summary"]["latency_ms"]]
    rr_note = (f"Runs execute in config order. Total latency is the CPU reranker's (rerank p50 {min(rr):.0f}–"
               f"{max(rr):.0f} ms across runs, scoring the same kind of candidates), so differences in the total "
               "column are reranker/CPU noise, not the index: compare the vector search column." if rr else "")
    prow = []
    for r in pooled["results"]:
        sp = r["spec"]
        label = index_label(sp) + (f" (nlist used {r['nlist_used']})" if "nlist_used" in r else "")
        prow.append([label, f"{r['build_s']:.2f}", f"{r['bytes'] / 2**20:.1f}",
                     f"{r['recall_vs_exact@10']['mean']:.3f} (min {r['recall_vs_exact@10']['min']:.2f})",
                     f"{r['recall_vs_exact@20']['mean']:.3f} (min {r['recall_vs_exact@20']['min']:.2f})",
                     f"{1000 * r['search_ms']['p50']:.0f} / {1000 * r['search_ms']['p95']:.0f}"])
    pooled_tab = md_table(["Index", "build s", "size MB", "recall vs exact @10", "recall vs exact @20",
                           "search µs p50 / p95"], prow)
    sizes = sorted(r["index_chunks"] for r in next(iter(runs.values()))["rows"].values())
    return f"""# S3 Results — Index types ({split} split)

**Generated** by `venv/bin/python -m evals.report --stage S3 --split {split}` from
`data/results/r4/s3/{split}/` (per-session runs: `venv/bin/python -m evals.run --config
evals/configs/s3_index_types.yaml`; pooled study: `venv/bin/python -m evals.index_study --config
evals/configs/s3_index_types.yaml`). Do not edit numbers by hand (F7). Parameters and scope:
`docs/18_session_log.md` (S-2026-11-22-33, S-2026-11-23-9).

**No pick** (§9.3): this stage shows the recall-vs-latency curve of approximate FAISS indexes against
exact search (`IndexFlatIP`, production). Configuration = the S2 winner, `C2` + `R-hybrid` + rerank; only
the dense list of R-hybrid uses the FAISS index. "Recall vs exact" (§8.3) = share of exact search's top k
that the approximate index's top k contains, per query, averaged.

## Per-question session indexes (as in every other stage)

Session index sizes over the {len(qids)} dev questions: {sizes[0]}–{sizes[-1]} chunks (median
{sizes[len(sizes) // 2]}). Latency per question; the total is dominated by the CPU reranker.

{session}

{rr_note}

## Pooled index: the whole dev corpus

{pooled['n_vectors']:,} `{pooled['chunker']}` vectors ({pooled['n_docs']} documents, each once, scans included;
d = {pooled['dim']}), {pooled['n_queries']} dev questions as queries. Search latency = median of
{pooled['repeats']} single-query searches per question, then p50 / p95 over questions (FAISS
{pooled['faiss_version']}, {pooled['search_threads']} thread — single-threaded in the evals, see `dense._faiss`). Top 20 is
what R-hybrid takes from the dense list.

{pooled_tab}

## Notes

- Index building is offline (as production builds a session's index at upload); build time and size are
  shown for the rollout question (R8: does the winner fit a 1 GB host?).
- The test split is untouched (§9.3).

## Provenance

Questions file sha256 `{pooled['questions_sha256'][:16]}…`, manifest `{pooled['manifest_sha256'][:16]}…`,
embedder `{pooled['embedder']}`. Pooled study git SHA `{pooled['git_sha'][:12]}`.

{md_table(["Run", "git SHA", "MLflow run", "seconds"],
          [[f"`{n}`", r["summary"]["git_sha"][:12], str(r["summary"].get("mlflow_run_id", "–"))[:12],
            str(r["summary"]["seconds"])] for n, r in runs.items()])}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", type=Path, help="default: CONFIGS[stage] in evals/configs/")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    config = a.config or REPO / "evals" / "configs" / CONFIGS.get(a.stage, "")
    cfg = load_config(config)
    runs = load_runs(a.stage, a.split, [r.name for r in cfg.runs], baseline=cfg.baseline)
    from evals.golden.qa.validate import QUESTIONS, read_questions
    from evals.run import file_sha256
    if file_sha256(QUESTIONS) != next(iter(runs.values()))["summary"]["questions_sha256"]:
        raise ValueError("questions file changed since the runs; re-run the stage before reporting")
    questions = {q["qid"]: q for q in read_questions(QUESTIONS)}
    out = a.out or DOCS.get((a.stage, a.split)) or REPO / "docs" / f"{a.stage.lower()}_{a.split}_retrieval.md"
    if a.stage == "S3":
        pooled = json.loads((OUT_ROOT / "s3" / a.split / "pooled" / "summary.json").read_text())
        if any(pooled[k] != next(iter(runs.values()))["summary"][k] for k in SAME if k in pooled):
            raise ValueError("pooled study and per-session runs used different setups; re-run both")
        out.write_text(render_s3(a.split, runs, pooled))
    else:
        out.write_text(render(a.stage, a.split, runs, questions))
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
