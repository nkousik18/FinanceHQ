"""§8.2 chunk metrics (evals/chunking/metrics.py) on small hand-built documents."""
import numpy as np
import pytest

from evals.chunking import metrics as M
from evals.chunking.base import Chunk, SourceDoc, Span, TableSource
from evals.chunking.c2_sentence_window import C2SentenceWindow
from evals.chunking.c3_fixed_tokens import C3FixedTokens
from evals.chunking.run_metrics import doc_metrics, summarize
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_count, tiktoken_count

P1 = "The Borrower shall repay the Loans. Interest accrues daily.\nSection 2.11 Prepayments apply."
#     0                                  35                       60
TABLE_PAGE = "Pricing Grid\nLevel Margin\nI 1.25%\nII 1.50%\nIII 1.75%"


def _chunk(i: int, spans: list[Span], text: str = "", doc: str = "d") -> Chunk:
    return Chunk(f"{doc}-{i}", doc, i, text or "x", spans)


def _doc() -> SourceDoc:
    return SourceDoc("d", {1: P1})


def _table_doc() -> SourceDoc:
    rows = ["Level Margin", "I 1.25%", "II 1.50%", "III 1.75%"]
    spans = []
    for r in rows:
        a = TABLE_PAGE.index(r)
        spans.append(Span(1, a, a + len(r)))
    md = "| Level | Margin |\n| --- | --- |\n| I | 1.25% |\n| II | 1.50% |\n| III | 1.75% |"
    ts = TableSource(1, 0, md, [r.split() for r in rows], spans)
    return SourceDoc("t", {1: TABLE_PAGE}, tables=[ts])


# ---- size ------------------------------------------------------------------------------------
def test_sizes_count_all_three_units():
    texts = ["Interest accrues daily.", "The applicable margin is 1.25% per annum."]
    sz = M.sizes([_chunk(i, [], t) for i, t in enumerate(texts)])
    assert sz.minilm == [minilm_count(t) for t in texts]
    assert sz.tiktoken == [tiktoken_count(t) for t in texts]
    assert sz.words == [3, 7]


# ---- truncation ------------------------------------------------------------------------------
def test_no_truncation_when_chunks_fit():
    src = _doc()
    chunks = [_chunk(0, [Span(1, 0, len(P1))], P1)]
    tr = M.truncation(src, chunks, M.sizes(chunks).minilm)
    assert (tr.truncated, tr.tokens_dropped, tr.page_words_never_embedded) == (0, 0, 0)
    assert tr.page_words == len(P1.split())


def test_words_outside_every_chunk_are_never_embedded():
    src = _doc()
    tr = M.truncation(src, [_chunk(0, [Span(1, 0, 35)], P1[:35])], [8])
    assert tr.page_words_never_embedded == len(P1[35:].split())


def test_truncated_chunk_embeds_only_its_head():
    words = [f"w{i}" for i in range(400)]
    page = " ".join(words)
    src = SourceDoc("d", {1: page})
    text = "### Page 1\n" + page                       # C1-style synthetic heading first
    n = minilm_count(text)
    assert n > MAX_CONTENT_TOKENS
    tr = M.truncation(src, [_chunk(0, [Span(1, 0, len(page))], text)], [n])
    cut = M.embedded_cut(text)
    kept = len(text[:cut].split()) - 3                   # minus "###", "Page", "1"
    assert tr.truncated == 1 and tr.tokens_dropped == n - MAX_CONTENT_TOKENS
    assert tr.unaligned_words == 0
    # the word straddling the cut counts as dropped
    assert tr.page_words_never_embedded in (400 - kept, 400 - kept + 1)


def test_overlap_rescues_a_truncated_tail():
    words = [f"w{i}" for i in range(400)]
    page = " ".join(words)
    src = SourceDoc("d", {1: page})
    tail_at = page.index("w300")
    chunks = [_chunk(0, [Span(1, 0, len(page))], page), _chunk(1, [Span(1, tail_at, len(page))], page[tail_at:])]
    tr = M.truncation(src, chunks, M.sizes(chunks).minilm)
    head = M.truncation(src, chunks[:1], M.sizes(chunks[:1]).minilm)
    assert tr.page_words_never_embedded < head.page_words_never_embedded


def test_table_row_words_align_through_markdown_pipes():
    pos = M._locate_words("| Level | Margin |\n| I | 1.25% |", [(0, 1, "I"), (2, 7, "1.25%")])
    assert pos == [21, 25]


def test_words_align_across_the_markdown_separator_row():
    text = "x\n[TABLE]\n| Section 4.12 | Financial Statements. | 33 |\n| --- | --- | --- |\n| Section 4.13 | Property. | 33 |"
    words = [(0, 0, w) for w in "Section 4.12 Financial Statements. 33 Section 4.13 Property. 33".split()]
    pos = M._locate_words(text, words)
    assert -1 not in pos and pos == sorted(pos)


def test_invisible_only_words_are_not_counted():
    src = SourceDoc("d", {1: "Interest \ufeff accrues"})
    tr = M.truncation(src, [_chunk(0, [Span(1, 0, 8), Span(1, 11, 18)])], [3])
    assert (tr.page_words, tr.page_words_never_embedded) == (2, 0)


# ---- tables ----------------------------------------------------------------------------------
def test_whole_table_in_one_chunk_is_not_split():
    src = _table_doc()
    tb = M.table_breaks(src, [_chunk(0, [Span(1, 0, len(TABLE_PAGE))])])
    assert (tb.tables, tb.split, tb.broken) == (1, 0, 0)


def test_split_without_header_is_broken():
    src = _table_doc()
    mid = TABLE_PAGE.index("II")
    tb = M.table_breaks(src, [_chunk(0, [Span(1, 0, mid - 1)]), _chunk(1, [Span(1, mid, len(TABLE_PAGE))])])
    assert (tb.split, tb.broken) == (1, 1)


def test_split_with_header_repeated_is_not_broken():
    src = _table_doc()
    t = src.tables[0]
    head = t.row_spans[0]
    c0 = _chunk(0, [Span(1, 0, t.row_spans[2].end)])
    c1 = _chunk(1, [head, t.row_spans[3]])                       # C4-style piece: header + rows
    c1b = _chunk(2, [head, Span(1, t.row_spans[3].end - 1, t.row_spans[3].end)])
    tb = M.table_breaks(src, [c0, c1, c1b])
    assert (tb.split, tb.broken) == (1, 0)


def test_table_without_header_span_is_skipped():
    src = _table_doc()
    src.tables[0].row_spans[0] = None
    assert M.table_breaks(src, [_chunk(0, [Span(1, 0, 5)])]).tables == 0


# ---- sentences -------------------------------------------------------------------------------
def test_sentence_edges_follow_lines_and_sentences():
    starts, ends = M.sentence_edges(P1)
    assert {0, 36, 60} <= starts
    assert {35, 59, len(P1)} <= ends


def test_sentence_aligned_chunks_have_no_breaks_and_doc_edges_are_excluded():
    sb = M.sentence_breaks(_doc(), [_chunk(0, [Span(1, 0, 35)]), _chunk(1, [Span(1, 36, len(P1))])])
    assert (sb.boundaries, sb.mid_sentence) == (2, 0)


def test_mid_sentence_cut_is_counted_on_both_sides():
    cut = P1.index("daily")
    sb = M.sentence_breaks(_doc(), [_chunk(0, [Span(1, 0, cut - 1)]), _chunk(1, [Span(1, cut, len(P1))])])
    assert (sb.boundaries, sb.mid_sentence) == (2, 2)


# ---- overlap redundancy ----------------------------------------------------------------------
def test_redundancy_zero_without_overlap_and_counts_repeats():
    src = _doc()
    tiled = [_chunk(0, [Span(1, 0, 35)]), _chunk(1, [Span(1, 36, len(P1))])]
    assert M.redundancy(src, tiled).duplicate_tokens == 0
    dup = M.redundancy(src, [_chunk(0, [Span(1, 0, len(P1))]), _chunk(1, [Span(1, 0, len(P1))])])
    assert dup.duplicate_tokens * 2 == dup.indexed_tokens


def test_spans_repeated_inside_one_chunk_count_once():
    src = _doc()
    r = M.redundancy(src, [_chunk(0, [Span(1, 0, 35), Span(1, 0, 35)])])
    assert r.duplicate_tokens == 0


# ---- evidence fragmentation ------------------------------------------------------------------
def test_locate_evidence_exact_and_whitespace_insensitive():
    src = _doc()
    assert M.locate_evidence(src, M.Evidence("d", 1, "Interest accrues daily.")) == Span(1, 36, 59)
    assert M.locate_evidence(src, M.Evidence("d", 1, "daily.  Section 2.11")) == Span(1, 53, 72)
    assert M.locate_evidence(src, M.Evidence("d", 1, "not there")) is None
    assert M.locate_evidence(src, M.Evidence("d", 9, "daily")) is None


def test_covers_needs_half_the_visible_chars_same_doc_and_page():
    src = _doc()
    ev = Span(1, 36, 59)                                    # "Interest accrues daily."
    assert M.covers(src, _chunk(0, [Span(1, 30, 59)]), ev)
    assert M.covers(src, _chunk(0, [Span(1, 36, 48)]), ev)       # 11 of 21 visible chars
    assert not M.covers(src, _chunk(0, [Span(1, 36, 44)]), ev)
    assert not M.covers(src, _chunk(0, [Span(1, 30, 59)], doc="other"), ev)


def test_fragmentation_counts_spans_no_single_chunk_covers():
    src = _doc()
    chunks = [_chunk(0, [Span(1, 0, 35)]), _chunk(1, [Span(1, 36, 59)]), _chunk(2, [Span(1, 60, len(P1))])]
    evs = [M.Evidence("d", 1, "The Borrower shall repay the Loans."),      # inside chunk 0
           M.Evidence("d", 1, P1),                                         # 3 chunks, none >= 50%
           M.Evidence("d", 1, "missing text"), M.Evidence("other", 1, "x")]
    fr = M.fragmentation(src, chunks, evs)
    assert (fr.spans, fr.located, fr.fragmented) == (3, 2, 1)


# ---- end to end on real chunkers --------------------------------------------------------------
LONG = ("Each Lender severally agrees to make Revolving Loans to the Borrower from time to time, "
        "in an aggregate amount not exceeding its Commitment, and ") * 12 + "so on."


@pytest.mark.parametrize("chunker", [C2SentenceWindow(), C3FixedTokens()], ids=lambda c: c.chunker_id)
def test_doc_metrics_on_real_chunkers(chunker):
    src = SourceDoc("d", {1: P1 + "\n" + LONG, 2: "After it. " + LONG})
    chunks = chunker.chunk(src)
    row, sz, vecs = doc_metrics(src, chunks, 0.0, embed=lambda ts: np.ones((len(ts), 4), np.float32))
    assert vecs.shape == (len(chunks), 4)
    assert row["truncated"] == 0 and row["page_words_never_embedded"] == 0
    assert row["duplicate_tokens"] > 0                    # both chunkers overlap
    if chunker.chunker_id == "C3":
        assert row["mid_sentence"] > 0                     # no sentence awareness
    s = summarize([row], [sz])
    assert s["truncation_rate_pct"]["point"] == 0
    assert s["chunks"] == len(chunks) and s["size_p5_p50_p95_max"]["minilm"][3] <= MAX_CONTENT_TOKENS


def test_r2_report_renders_from_summary():
    from evals.chunking.r2_report import render
    src = SourceDoc("d", {1: P1 + "\n" + LONG})
    chunks = C2SentenceWindow().chunk(src)
    row, sz, _ = doc_metrics(src, chunks, 0.0)
    s = summarize([row], [sz])
    s["by_part"] = {"edgar": summarize([row], [sz])}
    s["params"] = {"max_tokens": 200}
    s["index"] = {"vectors": len(chunks), "memory_mb": 0.1, "disk_mb": 0.1, "embed_seconds_per_doc": 0.5}
    md = render({"cleaning_version": "v2", "chunkers": {"C2": s}})
    assert "| ★ Truncation rate | 0% |" in md and "### edgar (1 docs)" in md and "pending R3" in md
