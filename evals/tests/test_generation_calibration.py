"""R6 judge calibration: question pick, frozen sample order, Cohen's κ (known answers), gate report,
labelling store. No network."""
import json
from collections import Counter

import pytest

from evals.generation import calibration as CAL
from evals.generation import run as GR
from evals.generation.calibration_app.server import LabelStore
from evals.tests.test_generation import FakeStore, Q, write_retrieval


def test_pick_questions_proportional_and_seeded():
    qs = [{"qid": f"q{t}{i}", "split": "dev", "type": t} for t, n in (("a", 10), ("b", 6), ("c", 4)) for i in range(n)]
    qs += [{"qid": "qt", "split": "test", "type": "a"}, {"qid": "qr", "split": "dev", "type": "a", "retired": True}]
    got = CAL.pick_questions(qs, 10)
    assert len(got) == 10 and got == CAL.pick_questions(qs, 10)
    assert Counter(q[1] for q in got) == {"a": 5, "b": 3, "c": 2}
    assert "qt" not in got and "qr" not in got


def ans(run, qid, typ, refused=False):
    return {"run": run, "qid": qid, "type": typ, "metrics": {"refused": refused}}


def test_order_sample_excludes_refusals_and_balances_prefix():
    pool = [ans(r, f"q{i}", t, refused=(i == 0)) for r in ("A", "B") for i, t in
            enumerate(["lookup", "table", "calc", "lookup", "table", "calc", "lookup", "table"])]
    order = CAL.order_sample(pool)
    assert len(order) == 14 and all(o["qid"] != "q0" for o in order)
    first6 = order[:6]
    assert Counter(o["run"] for o in first6) == {"A": 3, "B": 3}
    assert Counter(o["type"] for o in first6) == {"lookup": 2, "table": 2, "calc": 2}
    assert order == CAL.order_sample(pool)


def test_kappa_known_answers():
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert CAL.cohen_kappa(a, b, [True, False]) == pytest.approx(0.4)          # po .7, pe .5
    assert CAL.cohen_kappa(a, a, [True, False]) == pytest.approx(1.0)
    assert CAL.cohen_kappa([True] * 4, [True] * 4, [True, False]) is None        # chance agreement 1
    # linear weights: a one-step miss costs 1/4 on a 5-point scale
    assert CAL.cohen_kappa([1, 5], [2, 5], [1, 2, 3, 4, 5], "linear") == pytest.approx(1 - (0.25 / 2) / ((0.5 * 0.25 + 0.5 * 0.75 + 0.5 * 1 + 0.5 * 0) / 2))
    r = CAL.kappa_ci(a, b, [True, False])
    assert r["lo"] < 0.4 < r["hi"] and r["agreement"] == pytest.approx(0.7) and r["n"] == 50


def judgment(run, qid, corr, supported, rel=5):
    return {"run": run, "qid": qid, "judged_by": "llm",
            "grade": {"correctness": corr, "claims": [{"claim": "x", "supported": s} for s in supported],
                      "key_points": [], "relevance": rel}}


def test_report_gate_and_missing():
    keys = [("A", f"q{i}") for i in range(6)]
    corr = ["correct", "partial", "wrong", "correct", "wrong", "partial"]
    labels = {f"A|q{i}": {"correctness": c, "faithful": i % 2 == 0, "relevance": 5 - i % 3} for i, c in enumerate(corr)}
    judg = {("A", f"q{i}"): judgment("A", f"q{i}", c, [i % 2 == 0], 5 - i % 3) for i, c in enumerate(corr)}
    rows, missing = CAL.pair_up(keys, labels, judg)
    md = CAL.render(6, rows, missing, CAL.compute(rows), "judge-v1")
    assert "**Gate: PASSED**" in md and "INCOMPLETE" not in md
    del labels["A|q5"]
    judg[("A", "q0")] = judgment("A", "q0", "wrong", [False])
    rows, missing = CAL.pair_up(keys, labels, judg)
    md = CAL.render(6, rows, missing, CAL.compute(rows), "judge-v1")
    assert "NOT PASSED" in md and "A|q5: not labelled" in md and "`A|q0`" in md     # disagreement listed


def test_judge_no_claims_counts_as_faithful():
    assert CAL.judge_labels(judgment("A", "q", "correct", []))["faithful"] is True


def test_label_store_blind_item_and_validation(tmp_path):
    rroot, oroot = tmp_path / "r4", tmp_path / "r6"
    # retrieval for the dev split
    d = rroot / "s2" / "dev" / "A"
    write_retrieval(rroot, "S2", "A", ["q1"])
    d.parent.mkdir(parents=True)
    (rroot / "s2" / "test" / "A").rename(d)
    cfg = GR.GenConfig("S4", "m", 0.1, 1024, 10**9, [GR.GenRun("A", {"stage": "S2", "run": "A"})])
    out = oroot / "s4" / "dev" / "A"
    out.mkdir(parents=True)
    (out / "answers.jsonl").write_text(json.dumps({"qid": "q1", "answer": "Fee 2% [S2].", "run": "A"}) + "\n")
    qs = {"q1": {**Q, "question": "Fee?", "reference_answer": "2%.", "key_points": ["2%"]}}
    ls = LabelStore([("A", "q1")], qs, cfg, FakeStore(), tmp_path / "labels.json", "kousik",
                    out_root=oroot, retrieval_root=rroot)
    it = ls.item(0)
    assert [c["cited"] for c in it["context"]] == [False, True]
    assert "run" not in it and "metrics" not in it and "grade" not in json.dumps(it)   # blind
    with pytest.raises(ValueError):
        ls.save(0, {"correctness": "ok", "faithful": True, "relevance": 3})
    with pytest.raises(ValueError):
        ls.save(0, {"correctness": "correct", "faithful": True, "relevance": 7})
    ls.save(0, {"correctness": "partial", "faithful": False, "relevance": 4, "note": "n"})
    saved = json.loads((tmp_path / "labels.json").read_text())["labels"]["A|q1"]
    assert saved["correctness"] == "partial" and saved["labeled_by"] == "kousik"
    assert ls.list()[0]["labelled"] is True
