"""R4 step 4: results report (evals/report.py) on synthetic run folders."""
import json

import pytest

from evals import report as RP


def _write_run(root, name, recalls, entity=None, chunker="C1", limit=None, sha="abc", p50=10.0):
    d = root / "s1" / "dev" / name
    d.mkdir(parents=True)
    rows = []
    for i, r in enumerate(recalls):
        m = {k: None for k in RP.DIAGNOSTIC}
        m.update({"recall@budget": r, "entity_recall": entity[i] if entity else r, "mrr@10": r, "hit@10": r,
                  "context_words": 1000.0})
        rows.append({"qid": f"q{i:04d}", "metrics": m, "spans": 1, "unlocated": 0,
                     "fragmented": 0 if r else 1})
    (d / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mean = sum(recalls) / len(recalls)
    over = {k: {"mean": mean, "n": len(recalls)} for k in ("recall@budget", "entity_recall", "mrr@10", "hit@10")}
    over["context_words"] = {"mean": 1000.0, "n": len(recalls)}
    summary = {"run": {"name": name, "chunker": chunker, "retriever": "R-dense", "top_k": 20, "chunker_params": {}},
               "questions_sha256": sha, "manifest_sha256": "m" * 64, "embedder": "minilm", "table_config": "P2",
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
