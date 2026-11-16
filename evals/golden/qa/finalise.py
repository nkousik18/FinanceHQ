"""
R3 step 4: turn Kousik's reviewed drafts into the QA golden set (scope v2 §5.1, §5.3).

    venv/bin/python -m evals.golden.qa.finalise            # dry run: what would be written, checks
    venv/bin/python -m evals.golden.qa.finalise --write    # write evals/golden/qa/questions.jsonl

Input: `drafts/candidates.jsonl` + `drafts/reviews.json` (review app). Only reviews with decision
"keep" that are not stale are used; the question is the review's `final` (edited or as drafted),
`verified_by` / `verified_on` come from the review.

1. **Pick** exactly the §5.1 type quota from the kept questions (seeded, deterministic). Constraint
   items are taken first, so a random fill can't push them out: amendment cross_doc (≥ 10), scan
   evidence (≥ 8), consumer sessions (≥ 20%). The rest of each type's quota is a seeded random fill.
   Fewer kept than a quota → nothing is written; the shortfall is reported.
2. **Split** 40 dev / 80 test, stratified: each type's dev count by largest remainder (8/8/7/5/5/7),
   and within a type the items are ordered by stratum (consumer / scan / amendment / plain EDGAR)
   so dev and test get the same mix.
3. **qids** q0001… in type order, then candidate id. Each question keeps `source_cid`.
4. `schema.validate_set(final=True)` must pass, or nothing is written. An existing questions.jsonl is
   not overwritten without --force (after release, a fix is a new qid + the old one retired, F3).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

from evals.golden.qa.review_app.store import EDITABLE, draft_hash
from evals.golden.qa.schema import (MIN_AMENDMENT_QUESTIONS, MIN_CONSUMER_SHARE, MIN_SCAN_QUESTIONS,
                                    REQUIRED, SPLIT_SIZES, TYPE_COUNTS, TYPES, DocInfo, mix_report,
                                    validate_set)

log = logging.getLogger("qa_finalise")

HERE = Path(__file__).resolve().parent
DRAFTS = HERE / "drafts"
QUESTIONS = HERE / "questions.jsonl"
SEED = 20261116


def is_consumer(q: dict) -> bool:
    return q["session_id"].startswith("cfpb")          # same rule as schema.mix_report


def is_scan(q: dict, docs: dict[str, DocInfo]) -> bool:
    return any(docs.get(e["doc_id"]) and docs[e["doc_id"]].part == "simulated_scan" for e in q["evidence"])


def stratum(q: dict, docs: dict[str, DocInfo]) -> str:
    if is_scan(q, docs):
        return "scan"
    if is_consumer(q):
        return "consumer"
    return "amendment" if q["amendment"] else "edgar"


# ---- 0. kept questions -------------------------------------------------------------------------
def kept_questions(candidates: list[dict], reviews: dict[str, dict]) -> tuple[list[dict], dict]:
    """Questions from 'keep' reviews (not stale), in the golden-set shape without qid/split."""
    out, skipped = [], Counter()
    for c in candidates:
        r = reviews.get(c["cid"])
        if not r:
            skipped["not reviewed"] += 1
            continue
        if r.get("draft_hash") != draft_hash(c):
            skipped["stale review"] += 1
            continue
        if r["decision"] != "keep":
            skipped[r["decision"]] += 1
            continue
        q = {k: c[k] for k in ("type", "session_id", "session_docs", "amendment", "unanswerable", "provenance")}
        q.update({k: r["final"].get(k, c.get(k)) for k in EDITABLE})
        q.update({"verified_by": r["reviewed_by"], "verified_on": r["reviewed_on"],
                  "source_cid": c["cid"], "edited": bool(r.get("edited")), "drafted_by": c.get("drafted_by")})
        out.append(q)
    return out, dict(skipped)


# ---- 1. pick the quota ---------------------------------------------------------------------------
def pick(kept: list[dict], docs: dict[str, DocInfo], quotas: dict[str, int] = TYPE_COUNTS,
         seed: int = SEED, min_amend: int = MIN_AMENDMENT_QUESTIONS, min_scan: int = MIN_SCAN_QUESTIONS,
         min_consumer_share: float = MIN_CONSUMER_SHARE) -> tuple[list[dict], dict[str, int]]:
    """(chosen questions, shortfall per type). Constraint items first, then a seeded fill."""
    by_type = {t: sorted([q for q in kept if q["type"] == t], key=lambda q: q["source_cid"]) for t in quotas}
    short = {t: quotas[t] - len(qs) for t, qs in by_type.items() if len(qs) < quotas[t]}
    if short:
        return [], short
    rng = random.Random(seed)
    for qs in by_type.values():
        rng.shuffle(qs)
    chosen: dict[str, list[dict]] = {t: [] for t in quotas}

    def take(pred, n: int) -> None:
        for t in quotas:
            for q in by_type[t]:
                if n <= 0:
                    return
                if len(chosen[t]) < quotas[t] and q not in chosen[t] and pred(q):
                    chosen[t].append(q)
                    n -= 1

    take(lambda q: q["amendment"] and q["type"] == "cross_doc", min_amend)
    take(lambda q: is_scan(q, docs), min_scan - sum(is_scan(q, docs) for qs in chosen.values() for q in qs))
    need_consumer = math.ceil(round(min_consumer_share * sum(quotas.values()), 6))
    take(is_consumer, need_consumer - sum(is_consumer(q) for qs in chosen.values() for q in qs))
    for t in quotas:                                   # seeded fill
        for q in by_type[t]:
            if len(chosen[t]) >= quotas[t]:
                break
            if q not in chosen[t]:
                chosen[t].append(q)
    return [q for t in quotas for q in chosen[t]], {}


# ---- 2. split + 3. qids ------------------------------------------------------------------------
def dev_counts(type_counts: dict[str, int], dev_size: int) -> dict[str, int]:
    """Largest-remainder share of the dev split per type."""
    total = sum(type_counts.values())
    exact = {t: n * dev_size / total for t, n in type_counts.items()}
    out = {t: int(x) for t, x in exact.items()}
    for t in sorted(exact, key=lambda t: (-(exact[t] - out[t]), t))[:dev_size - sum(out.values())]:
        out[t] += 1
    return out


def split_and_number(chosen: list[dict], docs: dict[str, DocInfo], dev_size: int = SPLIT_SIZES["dev"],
                     seed: int = SEED) -> list[dict]:
    types = [t for t in TYPES if any(q["type"] == t for q in chosen)]
    counts = {t: sum(q["type"] == t for q in chosen) for t in types}
    want_dev = dev_counts(counts, dev_size)
    rng = random.Random(seed + 1)
    out = []
    for t in types:
        qs = sorted([q for q in chosen if q["type"] == t], key=lambda q: q["source_cid"])
        rng.shuffle(qs)
        qs.sort(key=lambda q: stratum(q, docs))         # stable: shuffled within each stratum
        n, d = len(qs), want_dev[t]
        # spread the d dev picks evenly over the stratum-ordered list
        dev_idx = {int((i + 0.5) * n / d) for i in range(d)} if d else set()
        for i, q in enumerate(qs):
            out.append({**q, "split": "dev" if i in dev_idx else "test"})
    out.sort(key=lambda q: (TYPES.index(q["type"]), q["source_cid"]))
    for i, q in enumerate(out, 1):
        q["qid"] = f"q{i:04d}"
    order = ["qid", "split", *[f for f in REQUIRED if f not in ("qid", "split")], "source_cid", "edited", "drafted_by"]
    return [{k: q[k] for k in order} for q in out]


# ---- main --------------------------------------------------------------------------------------
def finalise(candidates: list[dict], reviews: dict[str, dict], docs: dict[str, DocInfo], text) -> dict:
    kept, skipped = kept_questions(candidates, reviews)
    report = {"candidates": len(candidates), "kept": len(kept), "skipped": skipped,
              "kept_by_type": dict(Counter(q["type"] for q in kept))}
    chosen, short = pick(kept, docs)
    if short:
        return {**report, "shortfall": short, "questions": [], "errors": [f"type {t}: {n} short of quota" for t, n in short.items()]}
    qs = split_and_number(chosen, docs)
    errs = validate_set(qs, docs, text, final=True)
    return {**report, "unused_kept": len(kept) - len(qs), "mix": mix_report(qs, docs),
            "dev_by_type": dict(Counter(q["type"] for q in qs if q["split"] == "dev")),
            "edited": sum(q["edited"] for q in qs), "errors": errs, "questions": qs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write questions.jsonl (default: dry run)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing questions.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    from evals.golden.qa.validate import CacheText, corpus_docs
    docs = corpus_docs()
    cands = [json.loads(l) for l in (DRAFTS / "candidates.jsonl").read_text().splitlines() if l.strip()]
    revs_path = DRAFTS / "reviews.json"
    reviews = json.loads(revs_path.read_text()) if revs_path.exists() else {}
    out = finalise(cands, reviews, docs, CacheText(docs))
    qs = out.pop("questions")
    out["error_count"] = len(out["errors"])
    out["errors"] = out["errors"][:30]
    written = False
    if args.write and qs and not out["error_count"]:
        if QUESTIONS.exists() and QUESTIONS.read_text().strip() and not args.force:
            out["not_written"] = f"{QUESTIONS.name} exists; --force to overwrite (after release: new qid + retire instead)"
        else:
            QUESTIONS.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
            written = True
    out["written"] = str(QUESTIONS) if written else None
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if (qs and not out["error_count"]) else 1


if __name__ == "__main__":
    sys.exit(main())
