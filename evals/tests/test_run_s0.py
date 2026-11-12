"""Known-answer tests for the S0 decision and H10 rules (docs/20_s0_results.md §2)."""
from evals.ingest_metrics.run_s0 import decide, h10


def _rec(page, stratum, accs):
    return {"page_id": page, "stratum": stratum, "tables": {"per_table": [{"accuracy": a} for a in accs]}}


def _recs(p2_heavy, p2_other):
    p1 = [_rec(f"h{i}", "table_heavy", [0.5]) for i in range(10)] + \
         [_rec(f"o{i}", "complex_layout", [0.5]) for i in range(10)]
    p2 = [_rec(f"h{i}", "table_heavy", [p2_heavy]) for i in range(10)] + \
         [_rec(f"o{i}", "complex_layout", [p2_other]) for i in range(10)]
    return p1, p2


def test_h10_supported_when_p2_better_on_table_heavy():
    p1, p2 = _recs(p2_heavy=0.9, p2_other=0.1)
    out = h10(p1, p2)
    assert out["verdict"] == "supported" and out["n"] == 10


def test_h10_not_supported_when_p1_better():
    p1, p2 = _recs(p2_heavy=0.1, p2_other=0.9)
    assert h10(p1, p2)["verdict"] == "not supported"


def test_h10_inconclusive_on_tie():
    p1, p2 = _recs(p2_heavy=0.5, p2_other=0.9)
    assert h10(p1, p2)["verdict"] == "inconclusive"


def test_decide_uses_all_strata():
    p1, p2 = _recs(p2_heavy=0.9, p2_other=0.9)
    assert decide(p1, p2)["winner"] == "P2" and decide(p1, p2)["n"] == 20
