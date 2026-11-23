"""QA finalise: reviewed drafts -> golden set (evals/golden/qa/finalise.py). Synthetic corpus, no files."""
from collections import Counter

from evals.golden.qa import finalise as fz
from evals.golden.qa.review_app.store import EDITABLE, draft_hash
from evals.golden.qa.schema import TYPE_COUNTS, DictText, DocInfo

# a small synthetic corpus: 4 EDGAR deals (original + amendment + scan), 2 consumer sessions
DOCS, TEXTS = {}, {}
for d in range(1, 5):
    DOCS[f"deal_0{d}_ca"] = DocInfo(f"deal_0{d}_ca", "edgar", f"deal_0{d}", "original")
    DOCS[f"deal_0{d}_am1"] = DocInfo(f"deal_0{d}_am1", "edgar", f"deal_0{d}", "amendment_1")
    DOCS[f"scan_0{d}"] = DocInfo(f"scan_0{d}", "simulated_scan", f"deal_0{d}", f"scan_of:deal_0{d}_ca:p1")
    TEXTS[f"deal_0{d}_ca"] = {1: f"The commitment fee for deal {d} is 0.25% per annum. The Maturity Date is June 30, 2024."}
    TEXTS[f"deal_0{d}_am1"] = {1: f"For deal {d} the commitment fee is amended to 0.35% per annum."}
    TEXTS[f"scan_0{d}"] = {1: f"The commitment fee for deal {d} is 0.25% per annum."}
for s in ("cfpb_a", "cfpb_b"):
    DOCS[f"{s}_le"] = DocInfo(f"{s}_le", "consumer", s, "form")
    TEXTS[f"{s}_le"] = {1: "Estimated Closing Costs $5,099 Includes Loan Costs."}
TEXT = DictText(TEXTS)


def make(cid, qtype, session, ev, answer="0.25% per annum.", ent=("0.25%",), amendment=False, sup=(), num=None):
    docs = sorted(d for d, i in DOCS.items() if i.session == session)
    return {"cid": cid, "job_id": cid[:-2], "type": qtype, "question": f"Question {cid}?", "session_id": session,
            "session_docs": docs, "evidence": [{"doc_id": d, "page": 1, "span_text": t} for d, t in ev],
            "reference_answer": answer, "key_points": [answer], "entities": list(ent),
            "superseded_values": list(sup), "amendment": amendment, "answer_numeric": num, "numeric_tolerance": None,
            "unanswerable": qtype == "unanswerable", "provenance": "llm_drafted_human_verified",
            "drafted_by": "m", "drafted_on": "2026-11-16", "shown_pages": [[d, 1] for d, _ in ev], "draft_issues": []}


def pool(extra_per_type=3):
    """Enough synthetic candidates for every quota + constraint, with some slack."""
    cands, n = [], Counter()

    def add(qtype, **kw):
        n[qtype] += 1
        cands.append(make(f"{qtype}-{n[qtype]:03d}-a", qtype, **kw))
    fee = lambda d: (f"deal_0{d}_ca", "commitment fee for deal %d is 0.25%% per annum" % d)  # noqa: E731
    for i in range(TYPE_COUNTS["cross_doc"] + extra_per_type):
        d = i % 4 + 1
        add("cross_doc", session=f"deal_0{d}", ev=[(f"deal_0{d}_am1", "commitment fee is amended to 0.35% per annum"), fee(d)],
            answer="0.35% per annum.", ent=("0.35%",), amendment=True, sup=("0.25%",))
    for qtype in ("lookup", "table", "calculation", "explain"):
        for i in range(TYPE_COUNTS[qtype] + extra_per_type):
            num = 0.25 if qtype == "calculation" else None
            if i % 4 == 0:     # consumer
                add(qtype, session="cfpb_a" if i % 8 else "cfpb_b",
                    ev=[(f"{'cfpb_a' if i % 8 else 'cfpb_b'}_le", "Estimated Closing Costs $5,099")],
                    answer="$5,099.", ent=("$5,099",), num=5099 if num else None)
            elif i % 4 == 1:   # scan
                d = i % 4 + 1
                add(qtype, session=f"deal_0{d}", ev=[(f"scan_0{d}", "commitment fee for deal %d is 0.25%% per annum" % d)], num=num)
            else:
                d = i % 4 + 1
                add(qtype, session=f"deal_0{d}", ev=[fee(d)], num=num)
    for i in range(TYPE_COUNTS["unanswerable"] + extra_per_type):
        add("unanswerable", session=f"deal_0{i % 4 + 1}", ev=[], answer="The documents do not state this.", ent=())
    return cands


def keep_all(cands, edit=None):
    revs = {}
    for c in cands:
        final = {k: c[k] for k in EDITABLE}
        if edit and c["cid"] in edit:
            final.update(edit[c["cid"]])
        revs[c["cid"]] = {"decision": "keep", "final": final, "edited": bool(edit and c["cid"] in edit),
                          "reviewed_by": "kousik", "reviewed_on": "2026-11-17", "draft_hash": draft_hash(c)}
    return revs


def test_kept_questions_uses_final_and_skips_others():
    cands = pool(0)[:5]
    revs = keep_all(cands, edit={cands[0]["cid"]: {"question": "Edited?"}})
    revs[cands[1]["cid"]]["decision"] = "drop"
    revs[cands[2]["cid"]]["decision"] = "later"
    revs[cands[3]["cid"]]["draft_hash"] = "old"
    del revs[cands[4]["cid"]]
    kept, skipped = fz.kept_questions(cands, revs)
    assert [q["source_cid"] for q in kept] == [cands[0]["cid"]] and kept[0]["question"] == "Edited?"
    assert kept[0]["verified_by"] == "kousik" and kept[0]["edited"]
    assert skipped == {"drop": 1, "later": 1, "stale review": 1, "not reviewed": 1}


def test_dev_counts_real_quotas():
    d = fz.dev_counts(TYPE_COUNTS, 40)
    assert d == {"lookup": 8, "table": 8, "calculation": 7, "cross_doc": 5, "explain": 5, "unanswerable": 7}


def test_shortfall_writes_nothing():
    cands = [c for c in pool(0) if not c["cid"].startswith("unanswerable-001")]
    out = fz.finalise(cands, keep_all(cands), DOCS, TEXT)
    assert out["questions"] == [] and out["shortfall"] == {"unanswerable": 1}


def test_constraints_taken_before_fill():
    kept, _ = fz.kept_questions(pool(), keep_all(pool()))
    quotas = {"lookup": 2, "table": 2, "calculation": 2, "cross_doc": 2, "explain": 2, "unanswerable": 2}
    chosen, short = fz.pick(kept, DOCS, quotas, min_amend=2, min_scan=3, min_consumer_share=0.25)
    assert not short and len(chosen) == 12
    assert sum(fz.is_scan(q, DOCS) for q in chosen) >= 3
    assert sum(fz.is_consumer(q) for q in chosen) >= 3


def test_full_finalise_passes_validator_and_is_deterministic():
    cands = pool()
    revs = keep_all(cands)
    out = fz.finalise(cands, revs, DOCS, TEXT)
    assert out["errors"] == [], out["errors"][:5]
    qs = out["questions"]
    assert len(qs) == 120 and Counter(q["split"] for q in qs) == {"dev": 40, "test": 80}
    assert Counter(q["type"] for q in qs) == Counter(TYPE_COUNTS)
    assert [q["qid"] for q in qs] == [f"q{i:04d}" for i in range(1, 121)]
    assert out["unused_kept"] == len(cands) - 120
    assert fz.finalise(cands, revs, DOCS, TEXT)["questions"] == qs
    # dev gets each stratum, not just plain EDGAR
    dev = [q for q in qs if q["split"] == "dev"]
    assert {fz.stratum(q, DOCS) for q in dev} >= {"consumer", "scan", "amendment", "edgar"}


def test_fill_caps_consumer_for_edgar_share():
    # found 2026-11-22: the fill took 44/120 consumer items (EDGAR share < 70%); then a type filled early
    # used the cap and a later type with few EDGAR items was forced over it
    def q(cid, t, consumer):
        return {"source_cid": cid, "type": t, "session_id": "cfpb_a" if consumer else "deal_01",
                "amendment": False, "evidence": [{"doc_id": "x", "page": 1}]}
    kept = ([q(f"a-{i}", "lookup", i < 3) for i in range(5)] +      # lookup: 3 consumer, plenty EDGAR
            [q(f"b-{i}", "table", i < 2) for i in range(3)])         # table: only 1 EDGAR for 2 slots
    for seed in range(20):
        for min_consumer in (0.0, 0.25):        # the consumer minimum must not use up the cap either
            chosen, short = fz.pick(kept, {}, {"lookup": 2, "table": 2}, seed=seed, min_amend=0, min_scan=0,
                                    min_consumer_share=min_consumer, min_edgar_share=0.75)
            assert not short and sum(fz.is_consumer(x) for x in chosen) == 1, (seed, min_consumer)
