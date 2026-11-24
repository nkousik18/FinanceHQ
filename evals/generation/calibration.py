"""
Judge calibration (scope v2 §8.6; Kousik 2026-11-24: 30 answers now, 50 later; one faithfulness label
per answer). Dev split only — never test.

Steps (`python -m evals.generation.calibration <cmd>`):
1. `questions` — pick the dev questions to generate for, stratified by type (seeded), and write
   CAL_DIR/questions.json. Generate with `evals.generation.run --split dev --qids-file …`.
2. `sample`    — freeze the answers to label: non-refused dev answers (refusals are graded by rule, not
   by the judge), balanced across configurations and types, seeded. The ordered list is written once
   (CAL_DIR/sample.json); the first 30 are calibration v1, the next 20 extend it to 50 later. Frozen
   before the judge runs, so judge output can't steer the choice.
3. Kousik labels them blind in the app (`evals.generation.calibration_app.server`): correctness
   (correct/partial/wrong), faithful (every claim supported by the context: yes/no), relevance 1-5.
4. `report`    — Cohen's κ judge vs Kousik: correctness (3 classes, unweighted), faithfulness
   (binary; judge faithful = every judged claim supported, no claims = faithful), relevance (linearly
   weighted κ). Gate: κ >= 0.6 on correctness and faithfulness (relevance reported, not gating).
   Question-bootstrap 95% CIs. Writes docs/26_judge_calibration.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evals.generation import run as GR

REPO = Path(__file__).resolve().parents[2]
CAL_DIR = REPO / "evals" / "golden" / "judge_calibration"
QUESTIONS_FILE = CAL_DIR / "questions.json"
SAMPLE_FILE = CAL_DIR / "sample.json"
LABELS_FILE = CAL_DIR / "labels.json"
DOC = REPO / "docs" / "26_judge_calibration.md"
SEED = 20261124
GATE = 0.6
CORRECTNESS = ("correct", "partial", "wrong")


# ---- 1. questions ------------------------------------------------------------------------------
def pick_questions(questions: list[dict], n: int, seed: int = SEED) -> list[str]:
    """n dev qids, allocated to types in proportion to the dev mix (largest remainder), seeded."""
    dev = [q for q in questions if q["split"] == "dev" and not q.get("retired")]
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in dev:
        by_type[q["type"]].append(q["qid"])
    total = len(dev)
    quota = {t: n * len(v) / total for t, v in by_type.items()}
    alloc = {t: int(x) for t, x in quota.items()}
    for t in sorted(quota, key=lambda t: (alloc[t] - quota[t], t))[: n - sum(alloc.values())]:
        alloc[t] += 1
    rng = random.Random(seed)
    out = []
    for t in sorted(by_type):
        ids = sorted(by_type[t])
        rng.shuffle(ids)
        out += sorted(ids[: alloc[t]])
    return sorted(out)


# ---- 2. sample ---------------------------------------------------------------------------------
def order_sample(answers: list[dict], seed: int = SEED) -> list[dict]:
    """All non-refused answers in a seeded order that alternates configurations and spreads types:
    any prefix (30, 50) is as balanced as the pool allows."""
    pool = [a for a in answers if "error" not in a and not a["metrics"]["refused"]]
    rng = random.Random(seed)
    rng.shuffle(pool)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in pool:
        buckets[(a["run"], a["type"])].append(a)
    runs = sorted({a["run"] for a in pool})
    types = sorted({a["type"] for a in pool})
    out, seen_q = [], Counter()
    while any(buckets.values()):
        for r in runs:
            # least-used type first; prefer questions not yet sampled (spread over questions)
            cands = [(seen_t, t) for t in types if buckets[(r, t)]
                     for seen_t in [sum(1 for x in out if x["type"] == t)]]
            if not cands:
                continue
            _, t = min(cands)
            b = buckets[(r, t)]
            b.sort(key=lambda a: seen_q[a["qid"]])
            a = b.pop(0)
            seen_q[a["qid"]] += 1
            out.append(a)
    return [{"run": a["run"], "qid": a["qid"], "type": a["type"]} for a in out]


def sample_keys(n: int, path: Path = SAMPLE_FILE) -> list[tuple[str, str]]:
    items = json.loads(path.read_text())["order"]
    if len(items) < n:
        raise ValueError(f"sample has {len(items)} answers, {n} asked")
    return [(x["run"], x["qid"]) for x in items[:n]]


# ---- 4. statistics -----------------------------------------------------------------------------
def cohen_kappa(a: list, b: list, categories: list, weights: str | None = None) -> float | None:
    """Cohen's κ for two raters over the same items. weights: None (nominal) or 'linear' (ordinal,
    categories in order). None when undefined (no items, or chance agreement = 1)."""
    if not a:
        return None
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = (np.abs(i - j) / (k - 1)) if weights == "linear" else (i != j).astype(float)
    de = (w * exp).sum()
    if de == 0:
        return None
    return float(1 - (w * obs).sum() / de)


def kappa_ci(a: list, b: list, categories: list, weights: str | None = None, n: int = 5000,
             seed: int = 0) -> dict:
    point = cohen_kappa(a, b, categories, weights)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a))
        v = cohen_kappa([a[i] for i in ix], [b[i] for i in ix], categories, weights)
        if v is not None:
            boots.append(v)
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (None, None))
    agree = sum(x == y for x, y in zip(a, b)) / len(a) if a else None
    return {"kappa": point, "lo": None if lo is None else float(lo), "hi": None if hi is None else float(hi),
            "n": len(a), "agreement": agree}


def judge_labels(j: dict) -> dict:
    g = j["grade"]
    return {"correctness": g["correctness"], "faithful": all(c["supported"] for c in g["claims"]),
            "relevance": g["relevance"]}


def pair_up(keys: list[tuple[str, str]], labels: dict, judgments: dict) -> tuple[list, list[str]]:
    """[(key, human, judge)] for keys with both; and the missing keys with the reason."""
    rows, missing = [], []
    for run, qid in keys:
        k = f"{run}|{qid}"
        h, j = labels.get(k), judgments.get((run, qid))
        if h is None:
            missing.append(f"{k}: not labelled")
        elif j is None or j.get("grade") is None:
            missing.append(f"{k}: not judged")
        else:
            rows.append((k, h, judge_labels(j)))
    return rows, missing


def compute(rows) -> dict:
    h = [r[1] for r in rows]
    j = [r[2] for r in rows]
    return {
        "correctness": kappa_ci([x["correctness"] for x in h], [x["correctness"] for x in j], list(CORRECTNESS)),
        "faithfulness": kappa_ci([bool(x["faithful"]) for x in h], [x["faithful"] for x in j], [True, False]),
        "relevance": kappa_ci([int(x["relevance"]) for x in h], [int(x["relevance"]) for x in j],
                              [1, 2, 3, 4, 5], weights="linear"),
    }


def confusion(rows, key: str, cats: list) -> str:
    m = Counter((r[1][key], r[2][key]) for r in rows)
    head = "| Kousik ↓ / judge → | " + " | ".join(str(c) for c in cats) + " |"
    lines = [head, "|" + "---|" * (len(cats) + 1)]
    lines += [f"| {c} | " + " | ".join(str(m[(c, d)]) for d in cats) + " |" for c in cats]
    return "\n".join(lines)


def render(n: int, rows, missing: list[str], res: dict, judge_version: str) -> str:
    def cell(r):
        if r["kappa"] is None:
            return "–"
        ci = f" [{r['lo']:.2f}, {r['hi']:.2f}]" if r["lo"] is not None else ""
        return f"**{r['kappa']:.2f}**{ci}"
    gate = all(res[k]["kappa"] is not None and res[k]["kappa"] >= GATE for k in ("correctness", "faithfulness"))
    out = [f"# Judge calibration (generated by `evals.generation.calibration report`)", "",
           f"Judge rubric `{judge_version}`, `openai/gpt-oss-120b` (effort low, temperature 0). "
           f"Dev split only. Sample: first {n} of the frozen order (`{SAMPLE_FILE.relative_to(REPO)}`); "
           f"{len(rows)} paired.", ""]
    if missing:
        out += [f"> **INCOMPLETE — {len(missing)} of {n} missing:** " + "; ".join(missing[:8])
                + (" …" if len(missing) > 8 else ""), ""]
    out += ["| Metric | κ [95% CI] | Raw agreement | n | Gate (κ ≥ 0.6) |", "|---|---|---|---|---|"]
    for k, label, gating in (("correctness", "Correctness (3-class, Cohen's κ)", True),
                             ("faithfulness", "Faithfulness (all claims supported, Cohen's κ)", True),
                             ("relevance", "Relevance 1–5 (linear weighted κ)", False)):
        r = res[k]
        ok = "–" if not gating or r["kappa"] is None else ("pass" if r["kappa"] >= GATE else "**FAIL**")
        agree = "–" if r["agreement"] is None else f"{r['agreement']:.2f}"
        out.append(f"| {label} | {cell(r)} | {agree} | {r['n']} | {ok if gating else 'reported only'} |")
    out += ["", f"**Gate: {'PASSED' if gate and not missing else 'NOT PASSED'}** — judge numbers "
            + ("may be reported." if gate and not missing else "must not be reported yet (§8.6)."), ""]
    if rows:
        out += ["## Confusion matrices", "", "Correctness:", "", confusion(rows, "correctness", list(CORRECTNESS)), "",
                "Faithfulness:", "", confusion(rows, "faithful", [True, False]), "",
                "## Disagreements", "", "| Answer | Kousik | Judge |", "|---|---|---|"]
        for k, h, j in rows:
            if h["correctness"] != j["correctness"] or bool(h["faithful"]) != j["faithful"]:
                out.append(f"| `{k}` | {h['correctness']}, faithful {bool(h['faithful'])} | "
                           f"{j['correctness']}, faithful {j['faithful']} |")
    return "\n".join(out) + "\n"


def load_judgments(cfg, split: str = "dev", out_root: Path = GR.OUT_ROOT) -> dict:
    out = {}
    for r in cfg.runs:
        for j in GR.read_jsonl(GR.out_dir(cfg, r, split, out_root) / "judgments.jsonl"):
            if j.get("judged_by") == "llm":
                out[(j["run"], j["qid"])] = j          # last one wins (retries)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["questions", "sample", "report"])
    ap.add_argument("--config", type=Path, default=REPO / "evals" / "configs" / "s4_generation.yaml")
    ap.add_argument("--n-questions", type=int, default=24)
    ap.add_argument("--n", type=int, default=30, help="answers in this calibration round (30 now, 50 later)")
    ap.add_argument("--force", action="store_true", help="overwrite a frozen file")
    a = ap.parse_args(argv)
    cfg = GR.load_config(a.config)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "questions":
        from evals.golden.qa.validate import QUESTIONS, read_questions
        if QUESTIONS_FILE.exists() and not a.force:
            raise SystemExit(f"{QUESTIONS_FILE} exists (frozen); --force to overwrite")
        qids = pick_questions(read_questions(QUESTIONS), a.n_questions)
        QUESTIONS_FILE.write_text(json.dumps({"seed": SEED, "split": "dev", "qids": qids}, indent=1))
        print(f"{len(qids)} dev questions -> {QUESTIONS_FILE}")
    elif a.cmd == "sample":
        if SAMPLE_FILE.exists() and not a.force:
            raise SystemExit(f"{SAMPLE_FILE} exists (frozen); --force to overwrite")
        qids = set(json.loads(QUESTIONS_FILE.read_text())["qids"])
        answers = [x for r in cfg.runs for x in GR.read_jsonl(GR.out_dir(cfg, r, "dev") / "answers.jsonl")
                   if x["qid"] in qids]
        have = {(x["run"], x["qid"]) for x in answers if "error" not in x}
        need = {(r.name, q) for r in cfg.runs for q in qids
                if (GR.out_dir(cfg, r, "dev") / "answers.jsonl").exists()}
        if not need or need - have:
            raise SystemExit(f"dev generation incomplete ({len(need - have)} answers missing); "
                             "finish it before freezing the sample")
        order = order_sample(answers)
        SAMPLE_FILE.write_text(json.dumps({"seed": SEED, "pool": len(answers), "order": order}, indent=1))
        print(f"{len(order)} non-refused answers of {len(answers)} -> {SAMPLE_FILE}; "
              f"first {a.n}: {Counter(x['run'] for x in order[:a.n])}, {Counter(x['type'] for x in order[:a.n])}")
    else:
        from evals.generation.judge import JUDGE_VERSION
        keys = sample_keys(a.n)
        labels = json.loads(LABELS_FILE.read_text())["labels"] if LABELS_FILE.exists() else {}
        rows, missing = pair_up(keys, labels, load_judgments(cfg))
        res = compute(rows) if rows else {k: {"kappa": None, "lo": None, "hi": None, "n": 0, "agreement": None}
                                         for k in ("correctness", "faithfulness", "relevance")}
        DOC.write_text(render(a.n, rows, missing, res, JUDGE_VERSION))
        print(DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
