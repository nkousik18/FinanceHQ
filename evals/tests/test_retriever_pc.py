"""R5 step 2: R-pc (parent–child): section parents, window cap, fallback, dedupe, runner wiring."""
import json

import numpy as np
import pytest

from evals import run as R
from evals.chunking.base import Chunk, SourceDoc, Span
from evals.chunking.tokens import minilm_count
from evals.retrieval.index_cache import IndexCache
from evals.retrievers import ParentChildRetriever, SessionIndex
from evals.retrievers.parent_child import ParentMap, merged_spans
from evals.tests.test_retrieval_runner import FakeCorpus, _q, fake_embed

P1 = "Section 2.07 Fees. The Borrower shall pay a closing fee of $62,500 on the Closing Date."
P2 = "Section 6.12 Leverage. The Total Leverage Ratio shall not exceed 3.50 to 1.00."
LONG = " ".join(f"word{i} filler text here." for i in range(200))      # ~1,000 MiniLM tokens


def _doc(pages=None):
    return SourceDoc("d", pages or {1: P1, 2: P2})


def _c(cid, page, a, b, text, section="", doc="d"):
    return Chunk(cid, doc, 0, text, [Span(page, a, b)], section)


def _c4(src):
    """Fake C4: one section per page, two overlapping chunks each (like C4's windows)."""
    out = []
    for p, t in src.pages.items():
        mid = len(t) // 2
        out += [_c(f"s{p}a", p, 0, mid + 5, t[:mid + 5], f"Sec {p}"), _c(f"s{p}b", p, mid, len(t), t[mid:], f"Sec {p}")]
    return out


def _provenance_ok(src, chunk):
    return all(src.text_at(s) in chunk.text for s in chunk.spans)


# ---- helpers -----------------------------------------------------------------------------------
def test_merged_spans():
    assert merged_spans([Span(1, 5, 9), Span(1, 0, 6), Span(2, 0, 3), Span(1, 9, 12)]) == \
        [Span(1, 0, 12), Span(2, 0, 3)]


# ---- parents -----------------------------------------------------------------------------------
def test_child_gets_its_whole_small_section():
    src = _doc()
    pm = ParentMap({"d": src}, {"d": _c4(src)})
    par = pm.parent(_c("k", 1, 10, 30, P1[10:30]))
    assert par.text == "Sec 1\n" + P1 and par.spans == [Span(1, 0, len(P1))]
    assert par.meta["window"] is False and _provenance_ok(src, par)


def test_child_maps_to_section_with_most_overlap():
    src = _doc()
    pm = ParentMap({"d": src}, {"d": _c4(src)})
    child = Chunk("k", "d", 0, "x", [Span(1, len(P1) - 5, len(P1)), Span(2, 0, 30)])
    assert pm.parent(child).section == "Sec 2"


def test_no_section_falls_back_to_document_window():
    src = _doc()
    c4 = [_c("u", 1, 0, len(P1), P1, ""), _c("v", 2, 0, len(P2), P2, "Sec 2")]
    pm = ParentMap({"d": src}, {"d": c4})
    par = pm.parent(_c("k", 1, 0, 20, P1[:20]))
    assert "-PC-doc-" in par.chunk_id and par.section == ""
    assert par.text == P1 + "\n\n" + P2 and _provenance_ok(src, par)      # small doc: whole doc


def test_long_section_is_windowed_around_the_child():
    src = _doc({1: LONG})
    pm = ParentMap({"d": src}, {"d": [_c("s", 1, 0, len(LONG), LONG, "Big")]}, size=100)
    i = LONG.index("word100 ")
    par = pm.parent(_c("k", 1, i, i + 25, LONG[i:i + 25]))
    body = par.text[len("Big\n"):]
    assert par.meta["window"] is True and 95 <= minilm_count(body) <= 100
    assert "word100" in body and "word0 " not in body and _provenance_ok(src, par)
    # near the end: window clipped to the section, still full size
    j = LONG.index("word199")
    end = pm.parent(_c("k2", 1, j, len(LONG), LONG[j:]))
    assert end.text.endswith(LONG[-10:]) and minilm_count(end.text[len("Big\n"):]) >= 95


def test_same_window_is_one_parent_object():
    src = _doc()
    pm = ParentMap({"d": src}, {"d": _c4(src)})
    assert pm.parent(_c("a", 1, 0, 10, "x")) is pm.parent(_c("b", 1, 40, 60, "y"))


# ---- retriever ---------------------------------------------------------------------------------
def _session(src, children):
    idx = SessionIndex.build([(children, fake_embed([c.text for c in children]))])
    idx.extras["parents"] = ParentMap({"d": src}, {"d": _c4(src)})
    return idx


def test_pc_dedupes_parents_and_respects_k():
    src = _doc()
    kids = [_c("k0", 1, 0, 40, P1[:40]), _c("k1", 1, 40, len(P1), P1[40:]), _c("k2", 2, 0, len(P2), P2)]
    idx = _session(src, kids)
    q = "closing fee"
    hits = ParentChildRetriever().search(idx, fake_embed([q])[0], 10, query=q)
    assert [c.section for c in hits.chunks] == ["Sec 1", "Sec 2"]
    assert hits.scores == sorted(hits.scores, reverse=True) and hits.ranked(idx) is hits.chunks
    assert set(hits.timings) == {"vector_search", "parent_map"}
    assert len(ParentChildRetriever().search(idx, fake_embed([q])[0], 1, query=q).chunks) == 1


def test_pc_skips_child_already_inside_a_returned_window():
    src = _doc({1: LONG})
    i, j = LONG.index("word10 "), LONG.index("word12 ")
    a, b = _c("a", 1, i, i + 20, LONG[i:i + 20]), _c("b", 1, j, j + 20, LONG[j:j + 20])
    pm = ParentMap({"d": src}, {"d": [_c("s", 1, 0, len(LONG), LONG, "Big")]}, size=100)
    assert pm.parent(a).chunk_id != pm.parent(b).chunk_id       # different windows ...
    idx = SessionIndex.build([([a, b], fake_embed(["fee", "fee closing"]))])
    idx.extras["parents"] = pm
    hits = ParentChildRetriever().search(idx, fake_embed(["fee"])[0], 10)
    assert [c.chunk_id for c in hits.chunks] == [pm.parent(a).chunk_id]   # ... but b is inside a's


# ---- runner ------------------------------------------------------------------------------------
class FakeSections:
    """Stand-in parent chunker (C4 needs Docling layout elements): one section per page."""
    chunker_id, params = "FAKE-SEC", {}

    def chunk(self, doc):
        return [Chunk(f"{doc.doc_id}-S{p}", doc.doc_id, i, t, [Span(p, 0, len(t))], f"Page {p}")
                for i, (p, t) in enumerate(sorted(doc.pages.items()))]


def test_config_requires_parent_chunker_for_pc_only():
    with pytest.raises(ValueError):
        R.RunConfig("x", "C2", "R-pc", 5)
    with pytest.raises(ValueError):
        R.RunConfig("x", "C2", "R-dense", 5, parent_chunker="C4")
    assert R.RunConfig("x", "C2", "R-pc", 5, parent_chunker="C4").parent_chunker == "C4"


def test_run_stage_with_pc(tmp_path, monkeypatch):
    monkeypatch.setitem(R.CHUNKERS, "FAKE-SEC", FakeSections)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n")
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S2", "P2", [R.RunConfig("pc", "C3", "R-pc", 5, parent_chunker="FAKE-SEC")])
    s = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                    embed=fake_embed, cache=cache)["pc"]
    assert s["breakdown"]["overall"]["recall@budget"]["mean"] == 1.0
    assert set(s["latency_ms"]) == {"query_embed", "vector_search", "parent_map", "total"}
    row = json.loads((tmp_path / "out" / "s2" / "dev" / "pc" / "questions.jsonl").read_text())
    assert row["ranked"][0]["chunk_id"].startswith(("ca-PC-", "am-PC-"))
