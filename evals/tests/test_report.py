"""R4 step 4: results report (evals/report.py) on synthetic run folders."""
import json

import pytest

from evals import report as RP


def _write_run(root, name, recalls, entity=None, chunker="C1", limit=None, sha="abc", p50=10.0, stage="s1",
               retriever="R-dense", rerank=False, mrr=None, info=None, embedder="minilm"):
    d = root / stage / "dev" / name
    d.mkdir(parents=True)
    rows = []
    for i, r in enumerate(recalls):
        m = {k: None for k in RP.DIAGNOSTIC}
        m.update({"recall@budget": r, "entity_recall": entity[i] if entity else r,
                  "mrr@10": mrr[i] if mrr else r, "hit@10": r,
                  "context_words": 1000.0})
        rows.append({"qid": f"q{i:04d}", "metrics": m, "spans": 1, "unlocated": 0,
                     "fragmented": 0 if r else 1})
    (d / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mean = sum(recalls) / len(recalls)
    over = {k: {"mean": mean, "n": len(recalls)} for k in ("recall@budget", "entity_recall", "mrr@10", "hit@10")}
    over["context_words"] = {"mean": 1000.0, "n": len(recalls)}
    summary = {"run": {"name": name, "chunker": chunker, "retriever": retriever, "top_k": 20, "chunker_params": {},
                       "rerank": rerank, "parent_chunker": None},
               "retriever_info": info or {},
               "questions_sha256": sha, "manifest_sha256": "m" * 64, "embedder": embedder, "table_config": "P2",
               "cleaning_version": "v2", "git_sha": "f" * 40, "questions": len(recalls), "limit": limit,
               "cache": {"built": 0, "loaded": 1}, "seconds": 1.0,
               "latency_ms": {"total": {"p50": p50, "p95": 12.0}},
               "breakdown": {"overall": over, "type=lookup": over, "part=edgar": over}}
    (d / "summary.json").write_text(json.dumps(summary))


def test_report_ranks_compares_and_flags_ties(tmp_path):
    n = 40
    _write_run(tmp_path, "C1-dense", [0.0] * 20 + [1.0] * 20)                 # baseline 0.5
    _write_run(tmp_path, "C2-dense", [1.0] * 36 + [0.0] * 4, chunker="C2")    # 0.9: clearly better
    _write_run(tmp_path, "C3-dense", [1.0] * 35 + [0.0] * 5, chunker="C3")    # 0.875: tie with C2
    runs = RP.load_runs("S1", "dev", ["C1-dense", "C2-dense", "C3-dense"], root=tmp_path)
    md = RP.render("S1", "dev", runs)
    assert "1. `C2-dense` — recall@budget 0.900 ← **picked**" in md
    assert "2. `C3-dense` — recall@budget 0.875 ← **picked**" in md
    assert "3. `C1-dense` — recall@budget 0.500\n" in md
    assert "C2-dense better" in md                                        # vs baseline, CI excludes 0
    assert f"no detectable difference (n = {n})" in md                    # C2 vs C3
    assert "**0.500** [" in md and "type = lookup" in md and "part = edgar" in md
    assert "★ evidence fragmentation" in md and "**0.500** [" in md


def test_report_refuses_mixed_or_smoke_runs(tmp_path):
    _write_run(tmp_path, "a", [1.0, 0.0])
    _write_run(tmp_path, "b", [1.0, 0.0], sha="other")
    with pytest.raises(ValueError):
        RP.load_runs("S1", "dev", ["a", "b"], root=tmp_path)
    _write_run(tmp_path, "c", [1.0, 0.0], limit=2)
    with pytest.raises(ValueError):
        RP.load_runs("S1", "dev", ["a", "c"], root=tmp_path)
    with pytest.raises(FileNotFoundError):
        RP.load_runs("S1", "dev", ["a", "missing"], root=tmp_path)


def test_report_flags_a_tie_at_the_pick_boundary(tmp_path):
    _write_run(tmp_path, "C1-dense", [0.0] * 4 + [1.0] * 6)
    _write_run(tmp_path, "C2-dense", [1.0] * 9 + [0.0], chunker="C2")
    _write_run(tmp_path, "C5-dense", [0.0] * 2 + [1.0] * 8, chunker="C5", p50=9.0)   # faster, listed last
    _write_run(tmp_path, "C4-dense", [1.0] * 8 + [0.0] * 2, chunker="C4", p50=11.0)
    md = RP.render("S1", "dev", RP.load_runs("S1", "dev", ["C1-dense", "C2-dense", "C4-dense", "C5-dense"],
                                             root=tmp_path))
    assert "**Tie at the pick boundary:** `C5-dense`, `C4-dense`" in md and "faster wins" in md
    assert "2. `C5-dense` — recall@budget 0.800 ← **picked**" in md
    assert "Evidence fragmentation is 0" not in md                      # C1 has fragmented rows


# ---- S2 (R5 step 4) ----------------------------------------------------------------------------
@pytest.mark.parametrize("q, ok", [
    ("What is the Applicable Rate?", True), ("Under Section 2.11, what applies?", True),
    ("Under clause (o), who is excluded?", True), ("What does 'Eligible Accounts' exclude?", True),
    ("What is the fee in 2020?", True), ("When does the rate lock expire for this loan?", False),
    ("Who is the borrower's agent?", False), ("Which bank is the agent?", False)])
def test_has_identifier(q, ok):
    assert RP.has_identifier(q) is ok


def test_overall_verdict_rule():
    assert RP._overall(["above", "above"]) == "**supported**"
    assert RP._overall(["above", "includes 0"]) == "**inconclusive**"
    assert RP._overall(["above", "below"]) == "**not supported**"
    assert RP._overall([]).startswith("not tested")


def test_external_baseline_loaded_first_and_not_picked(tmp_path):
    _write_run(tmp_path, "C1-dense", [1.0] * 10, stage="s1")                  # best, but external
    _write_run(tmp_path, "C2-dense", [1.0] * 5 + [0.0] * 5, chunker="C2", stage="s2")
    _write_run(tmp_path, "C2-bm25", [1.0] * 6 + [0.0] * 4, chunker="C2", stage="s2", retriever="R-bm25")
    runs = RP.load_runs("S2", "dev", ["C2-dense", "C2-bm25"], root=tmp_path,
                        baseline={"stage": "S1", "run": "C1-dense"})
    assert list(runs) == ["C1-dense", "C2-dense", "C2-bm25"] and runs["C1-dense"]["external"]
    assert RP.rank_order(runs, RP.headline(runs, sorted(runs["C1-dense"]["rows"]))[1]) == ["C2-bm25", "C2-dense"]


def test_baseline_with_a_different_setup_is_refused(tmp_path):
    _write_run(tmp_path, "C1-dense", [1.0] * 4, stage="s1", embedder="other")
    _write_run(tmp_path, "C2-dense", [1.0] * 4, chunker="C2", stage="s2")
    with pytest.raises(ValueError):
        RP.load_runs("S2", "dev", ["C2-dense"], root=tmp_path, baseline={"stage": "S1", "run": "C1-dense"})


def test_s2_report_hypotheses_latency_and_truncation(tmp_path):
    n = 20
    up = [1.0] * n
    _write_run(tmp_path, "C1-dense", [0.5] * n, stage="s1")
    _write_run(tmp_path, "C2-dense", [0.0] * n, chunker="C2", stage="s2", mrr=[0.2] * n)
    _write_run(tmp_path, "C2-hybrid", up, chunker="C2", stage="s2", retriever="R-hybrid", mrr=[0.2] * n)
    _write_run(tmp_path, "C2-dense+rerank", up, chunker="C2", stage="s2", rerank=True, mrr=[0.9] * n,
               info={"rerank_candidates": 400, "rerank_truncated": 40})
    runs = RP.load_runs("S2", "dev", ["C2-dense", "C2-hybrid", "C2-dense+rerank"], root=tmp_path,
                        baseline={"stage": "S1", "run": "C1-dense"})
    qs = {f"q{i:04d}": {"type": "lookup", "question": "What is the Applicable Rate?" if i < 10 else "What fee?"}
          for i in range(n)}
    md = RP.render("S2", "dev", runs, qs)
    assert "Subset: 10 of 20 dev lookup questions" in md
    h2, h3 = md.split("**H3**")
    assert "`C2-hybrid` − `C2-dense` | 10 | +1.000" in h2 and "Verdict (dev): **supported**" in h2
    assert "`C2-dense+rerank` − `C2-dense` | 20 | +0.700" in h3 and "Verdict (dev): **supported**" in h3
    assert "| `C2-dense+rerank` | 400 | 40 | 10.0% |" in md
    assert "total p50 / p95" in md and "C1-dense (S1)" in md and "Baseline `C1-dense` is the S1 run" in md
    assert "1. `C2-hybrid`" in md and "`C1-dense` —" not in md.split("## Pick")[1].split("##")[0]
    assert "R-dense+rerank" in md
