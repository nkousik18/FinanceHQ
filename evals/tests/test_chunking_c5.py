"""C5 (semantic): boundaries at distance spikes, C2 token bounds, exact provenance, coverage."""
import numpy as np
import pytest

from evals.chunking.base import SourceDoc
from evals.chunking.c2_sentence_window import sentence_ranges
from evals.chunking.c5_semantic import C5Semantic, _merge_small, _split_over_max, adjacent_distances, semantic_cuts
from evals.chunking.stream import build_stream
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_count
from evals.tests.test_chunking_c2_c3 import _doc


def topic_embedder(texts):
    """Unit vector per topic: 'rate' sentences vs 'collateral' sentences vs everything else."""
    out = []
    for t in texts:
        low = t.lower()
        v = [1.0, 0, 0] if "rate" in low else [0, 1.0, 0] if "collateral" in low else [0, 0, 1.0]
        out.append(v)
    return np.array(out)


RATE = "The interest rate is the Base Rate plus the Applicable Margin of two percent per annum."
COLL = "The collateral includes all inventory, receivables and equipment of each Loan Party."


def _two_topics() -> SourceDoc:
    return SourceDoc("t", {1: " ".join([RATE] * 4) + "\n" + " ".join([COLL] * 4)})


def test_cut_at_topic_shift():
    src = _two_topics()
    chunks = C5Semantic(min_tokens=10, embedder=topic_embedder).chunk(src)
    assert [c.text.count("rate") > 0 for c in chunks] == [True, False]
    assert chunks[1].meta["boundary"] == "semantic"
    assert chunks[1].text.startswith("The collateral")


def _all_chunkers():
    return [C5Semantic(embedder=topic_embedder), C5Semantic(max_tokens=60, min_tokens=15,
                                                            embedder=topic_embedder)]


@pytest.mark.parametrize("chunker", _all_chunkers(), ids=lambda c: str(c.params["max_tokens"]))
def test_spans_exact_tokens_fit_and_full_coverage(chunker):
    src = _doc()
    chunks = chunker.chunk(src)
    assert chunks
    covered = {p: [False] * len(t) for p, t in src.pages.items()}
    for c in chunks:
        n = minilm_count(c.text)
        assert n == c.meta["minilm_tokens"] <= chunker.params["max_tokens"] <= MAX_CONTENT_TOKENS
        assert c.spans
        for s in c.spans:
            piece = src.text_at(s)
            assert piece and piece.strip() == piece and piece in c.text
            covered[s.page][s.start:s.end] = [True] * (s.end - s.start)
    for p, t in src.pages.items():
        assert all(ok or ch.isspace() for ok, ch in zip(covered[p], t)), p


def test_no_overlap_and_sentence_aligned_boundaries():
    src = _doc()
    sents = sentence_ranges(build_stream(src).text)
    starts, ends = {a for a, _ in sents}, {b for _, b in sents}
    chunks = C5Semantic(max_tokens=60, min_tokens=15, embedder=topic_embedder).chunk(src)
    for a, b in zip(chunks, chunks[1:]):
        assert a.meta["stream_range"][1] <= b.meta["stream_range"][0]
    for c in chunks:
        if not c.meta["split_sentence"]:
            a, b = c.meta["stream_range"]
            assert a in starts and b in ends


def test_adjacent_distances():
    e = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
    assert adjacent_distances(e).tolist() == [0.0, 1.0]
    assert adjacent_distances(e[:1]).size == 0


def test_split_over_max_cuts_at_largest_distance():
    tok = np.arange(0, 110, 10)                      # 10 units of 10 tokens
    dist = np.array([.1, .1, .9, .1, .1, .5, .1, .1, .1])   # dist[k-1] before unit k
    assert _split_over_max(0, 10, tok, dist, 100) == []
    assert _split_over_max(0, 10, tok, dist, 50) == [3, 6]   # 3 then the 7-unit side at 6
    assert all(b - a <= 5 for a, b in zip([0, 3, 6], [3, 6, 10]))


def test_merge_small_prefers_more_similar_neighbour_and_respects_max():
    tok = np.array([0, 40, 45, 85])                  # sizes 40, 5, 40
    dist = np.array([.8, .2])                       # small unit is closer to the right one
    assert _merge_small([1, 2], 3, tok, dist, 10, 100) == [1]
    assert _merge_small([1, 2], 3, tok, dist, 10, 44) == [1, 2]   # neither merge fits


def test_semantic_cuts_threshold_is_document_percentile():
    tok = np.arange(0, 210, 10)                      # 20 units, 10 tokens each
    dist = np.full(19, .1)
    dist[9] = .9                                     # before unit 10
    cuts, thr = semantic_cuts(dist, tok, 95, 20, 500)
    assert cuts == [10] and .1 <= thr < .9


def test_tiny_and_empty_docs():
    ch = C5Semantic(embedder=topic_embedder)
    assert ch.chunk(SourceDoc("e", {1: "   "})) == []
    one = ch.chunk(SourceDoc("o", {1: "Only one sentence."}))
    assert len(one) == 1 and one[0].meta["doc_threshold"] is None
    with pytest.raises(ValueError):
        C5Semantic(min_tokens=200, embedder=topic_embedder)


def test_real_minilm_embedder_separates_topics():
    chunks = C5Semantic(min_tokens=10).chunk(_two_topics())
    assert len(chunks) == 2 and chunks[1].text.startswith("The collateral")
