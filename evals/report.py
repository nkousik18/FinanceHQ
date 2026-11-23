"""
Results report for the retrieval stages (scope v2 §9.2 report.py, §9.6 statistics). Numbers are
generated from the run folders, never typed by hand (F7).

    venv/bin/python -m evals.report --stage S1 --split dev        # -> docs/22_s1_retrieval.md

Reads data/results/r4/<stage>/<split>/<run>/{summary.json, questions.jsonl} for every run in the
stage config, in config order. All runs must have scored the same questions file (sha256) and the
same question ids; otherwise the report refuses (no mixed comparisons).

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
import sys
from pathlib import Path

from evals.run import OUT_ROOT, REPO, load_config
from evals.stats import bootstrap_ci, holm, paired_diff_ci, ratio_ci

STARRED = ("recall@budget", "entity_recall")
PICKS = {"S1": 2, "S2": 3}                     # §9.3: S1 top 2 chunkers, S2 top 3 configurations
DOCS = {("S1", "dev"): REPO / "docs" / "22_s1_retrieval.md"}
TYPES = ("lookup", "table", "calculation", "cross_doc", "explain", "unanswerable")
PARTS = ("edgar", "consumer", "simulated_scan")
DIAGNOSTIC = ("recall@1", "recall@3", "recall@5", "recall@10", "hit@1", "hit@5", "hit@10",
              "precision@1", "precision@5", "precision@10", "context_precision", "mrr@10", "ndcg@10",
              "map@10", "noise_ratio", "redundancy", "context_words", "cross_doc_coverage",
              "amend_coverage", "amend_only_superseded")


# ---- loading -----------------------------------------------------------------------------------
def load_runs(stage: str, split: str, names: list[str], root: Path = OUT_ROOT) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for name in names:
        d = root / stage.lower() / split / name
        if not (d / "summary.json").exists():
            raise FileNotFoundError(f"run {name}: no results in {d} (run evals.run first)")
        summary = json.loads((d / "summary.json").read_text())
        rows = [json.loads(line) for line in (d / "questions.jsonl").open()]
        runs[name] = {"summary": summary, "rows": {r["qid"]: r for r in rows}}
    shas = {r["summary"]["questions_sha256"] for r in runs.values()}
    qids = {tuple(sorted(r["rows"])) for r in runs.values()}
    if len(shas) > 1 or len(qids) > 1:
        raise ValueError("runs scored different question sets; re-run the stage before reporting")
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
        rows.append([name, rc["chunker"], rc["retriever"], str(rc["top_k"]), ci_cell(s["recall@budget"]),
                     ci_cell(s["entity_recall"]), f3(o["mrr@10"]["mean"]), f3(o["hit@10"]["mean"]),
                     f"{o['context_words']['mean']:.0f}", ci_cell(s["fragmentation"]),
                     f"{lat['p50']:.1f} / {lat['p95']:.1f}"])
    n = stats[next(iter(stats))]["recall@budget"]["n"]
    table = md_table(["Run", "Chunker", "Retriever", "top_k", f"★ recall@budget (n={n})", "★ entity recall",
                      "mrr@10", "hit@10", "context words", "★ evidence fragmentation", "retrieval ms p50 / p95"],
                     rows)
    return table, stats


def ranking(runs: dict, stats: dict, stage: str) -> tuple[str, list[str]]:
    order = sorted(runs, key=lambda r: -(stats[r]["recall@budget"]["point"] or 0))
    k = PICKS.get(stage, 2)
    lines = [f"{i}. `{r}` — recall@budget {stats[r]['recall@budget']['point']:.3f}" +
             (" ← **picked**" if i <= k else "") for i, r in enumerate(order, 1)]
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


def render(stage: str, split: str, runs: dict) -> str:
    qids = sorted(next(iter(runs.values()))["rows"])
    head_table, stats = headline(runs, qids)
    rank_text, _ = ranking(runs, stats, stage)
    order = sorted(runs, key=lambda r: -(stats[r]["recall@budget"]["point"] or 0))
    base = next(iter(runs))
    types = [(f"type = {t}", f"type={t}") for t in TYPES]
    parts = [(f"part = {p}", f"part={p}") for p in PARTS]
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

## Provenance

{provenance(runs)}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", type=Path, help="default: evals/configs/s1_chunkers.yaml for S1")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    config = a.config or REPO / "evals" / "configs" / {"S1": "s1_chunkers.yaml"}.get(a.stage, "")
    names = [r.name for r in load_config(config).runs]
    runs = load_runs(a.stage, a.split, names)
    out = a.out or DOCS.get((a.stage, a.split)) or REPO / "docs" / f"{a.stage.lower()}_{a.split}_retrieval.md"
    out.write_text(render(a.stage, a.split, runs))
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
