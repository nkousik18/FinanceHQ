"""C2 (sentence window) and C3 (fixed tokens): provenance (F1), token limits, coverage."""
import pytest

from evals.chunking.base import SourceDoc, Span
from evals.chunking.c2_sentence_window import C2SentenceWindow, _Unit, pack, sentence_ranges
from evals.chunking.c3_fixed_tokens import C3FixedTokens, token_windows
from evals.chunking.source import build_source
from evals.chunking.stream import build_stream
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_count, minilm_offsets, tiktoken_count
from evals.tests.test_chunking_c1 import _extraction

LONG = ("Each Lender severally agrees to make Revolving Loans to the Borrower from time to time, "
        "in an aggregate amount not exceeding its Commitment, and ") * 12 + "so on."


def _doc() -> SourceDoc:
    src = build_source(_extraction())
    src.pages[4] = "ARTICLE III\nLong sentence follows. " + LONG + " After it. Section 2.11(a) applies."
    return src


CHUNKERS = [C2SentenceWindow(), C2SentenceWindow(60, 15), C3FixedTokens(), C3FixedTokens(64, 16)]


def _limit(ch):
    return ch.params.get("max_tokens") or ch.params["content_tokens"]


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: f"{c.chunker_id}-{c.params}")
def test_f1_spans_are_exact_substrings(chunker):
    src = _doc()
    chunks = chunker.chunk(src)
    assert chunks
    for c in chunks:
        assert c.spans, c.index
        for s in c.spans:
            piece = src.text_at(s)
            assert piece.strip() == piece and piece, (c.index, s)
            assert piece in c.text
        assert c.page_start <= c.page_end


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: f"{c.chunker_id}-{c.params}")
def test_chunks_fit_their_token_limit(chunker):
    for c in chunker.chunk(_doc()):
        n = minilm_count(c.text)
        assert n == c.meta["minilm_tokens"], c.index        # stream count == re-tokenized count
        assert n <= _limit(chunker) <= MAX_CONTENT_TOKENS


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: f"{c.chunker_id}-{c.params}")
def test_every_page_character_is_covered(chunker):
    src = _doc()
    covered = {p: [False] * len(t) for p, t in src.pages.items()}
    for c in chunker.chunk(src):
        for s in c.spans:
            covered[s.page][s.start:s.end] = [True] * (s.end - s.start)
    for p, t in src.pages.items():
        assert all(ok or ch.isspace() for ok, ch in zip(covered[p], t)), p


def test_c2_breaks_only_at_sentence_boundaries_unless_split():
    src = _doc()
    sents = sentence_ranges(build_stream(src).text)
    starts, ends = {a for a, _ in sents}, {b for _, b in sents}
    chunks = C2SentenceWindow(60, 15).chunk(src)
    assert any(c.meta["split_sentence"] for c in chunks)
    for c in chunks:
        if not c.meta["split_sentence"]:
            a, b = c.meta["stream_range"]
            assert a in starts and b in ends, c.index


def test_sentence_ranges_legal_text_and_lines():
    text = "Section 2.11(a). Acme, Inc. and U.S. Bank, N.A. agree.\nARTICLE II LOANS\n\nNext one."
    got = [text[a:b] for a, b in sentence_ranges(text)]
    assert got == ["Section 2.11(a).", "Acme, Inc. and U.S. Bank, N.A. agree.", "ARTICLE II LOANS",
                   "Next one."]


def test_pack_overlap_and_max():
    units = [_Unit(i * 10, i * 10 + 9, i * 10, i * 10 + 10) for i in range(10)]   # 10 tokens each
    groups = pack(units, 35, 12)
    assert [[u.ti for u in g] for g in groups] == [[0, 10, 20], [20, 30, 40], [40, 50, 60],
                                                   [60, 70, 80], [80, 90]]
    assert all(g[-1].tj - g[0].ti <= 35 for g in groups)
    no_overlap = pack(units, 35, 5)                          # a 10-token sentence > 5: no overlap
    assert [[u.ti for u in g] for g in no_overlap] == [[0, 10, 20], [30, 40, 50], [60, 70, 80], [90]]


def test_token_windows_respect_words_and_overlap():
    word_ids = [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7]         # e.g. "pre ##pay" = word 0
    wins = token_windows(word_ids, 0, len(word_ids), 4, 1)
    for i, j in wins:
        assert j - i <= 4
        assert i == 0 or word_ids[i] != word_ids[i - 1]
        assert j == len(word_ids) or word_ids[j] != word_ids[j - 1]
    assert wins[0][0] == 0 and wins[-1][1] == len(word_ids)
    assert all(b[0] < a[1] or b[0] == a[1] for a, b in zip(wins, wins[1:]))   # no gaps
    with pytest.raises(ValueError):
        token_windows(word_ids, 0, 5, 2, 2)


def test_c3_overlap_is_at_least_requested():
    chunks = C3FixedTokens(64, 16).chunk(_doc())
    for a, b in zip(chunks, chunks[1:]):
        assert a.meta["token_range"][1] - b.meta["token_range"][0] >= 16


def test_stream_spans_skip_page_separator():
    src = SourceDoc("d", {1: "abc ", 3: "  xyz"})
    st = build_stream(src)
    assert st.spans(0, len(st.text)) == [Span(1, 0, 3), Span(3, 2, 5)]
    assert st.spans(3, 6) == []


def test_token_counters():
    offsets, word_ids = minilm_offsets("Prepayment at 2.50%.")
    assert len(offsets) == minilm_count("Prepayment at 2.50%.") == len(word_ids)
    assert tiktoken_count("Prepayment at 2.50%.") > 0


@pytest.mark.parametrize("chunker", [C2SentenceWindow(), C3FixedTokens()], ids=lambda c: c.chunker_id)
def test_invisible_format_chars_never_edge_or_fill_a_span(chunker):
    src = SourceDoc("d", {1: "﻿First sentence here.\n﻿\nSecond one.﻿", 2: "﻿"})
    chunks = chunker.chunk(src)
    spans = [s for c in chunks for s in c.spans]
    assert spans and all(src.trim(s) == s for s in spans)
    assert all(s.page == 1 for s in spans)


def test_c1_trims_bom_lines():
    from evals.chunking.c1_line_window import C1LineWindow, _page_lines
    assert _page_lines(1, "﻿\nabc")[0].spans == []
    er = _extraction()
    er.pages[0].text = "﻿ARTICLE I " + er.pages[0].text
    src = build_source(er)
    assert all(src.trim(s) == s for c in C1LineWindow().chunk(src) for s in c.spans)
