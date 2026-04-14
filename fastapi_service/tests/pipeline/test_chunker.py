"""
Tests for app/pipeline/chunker.py
Run: pytest fastapi_service/tests/pipeline/test_chunker.py -v
"""
import pytest
from app.pipeline.chunker import (
    chunk_document, Chunk,
    _word_count, _extract_section, _extract_page_from_heading, _slide,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from app.pipeline.cleaner import CleanedResult, CleanedPage, CleanedFormField, CleanedTable


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_cleaned(
    session_id: str = "sess-001",
    page_texts: list[str] | None = None,
    form_fields: list[CleanedFormField] | None = None,
    tables: list[CleanedTable] | None = None,
) -> CleanedResult:
    pages = [
        CleanedPage(page_number=i + 1, text=t, avg_confidence=98.0, has_handwriting=False)
        for i, t in enumerate(page_texts or ["Sample loan text on page one."])
    ]
    return CleanedResult(
        session_id=session_id,
        pages=pages,
        form_fields=form_fields or [],
        tables=tables or [],
        total_pages=len(pages),
        overall_avg_confidence=98.0,
        has_handwriting=False,
    )


def _make_field(key: str, value: str) -> CleanedFormField:
    return CleanedFormField(key=key, value=value, page=1, confidence=95.0)


# ------------------------------------------------------------------
# Unit helpers
# ------------------------------------------------------------------

class TestWordCount:
    def test_counts_words(self):
        assert _word_count("hello world foo") == 3

    def test_empty_string(self):
        assert _word_count("") == 0

    def test_single_word(self):
        assert _word_count("loan") == 1


class TestExtractSection:
    def test_h2_heading(self):
        assert _extract_section("## Form Fields") == "Form Fields"

    def test_h3_heading(self):
        assert _extract_section("### Page 3") == "Page 3"

    def test_non_heading(self):
        assert _extract_section("Loan Amount: 750000") is None

    def test_empty_line(self):
        assert _extract_section("") is None


class TestExtractPageFromHeading:
    def test_detects_page_number(self):
        assert _extract_page_from_heading("Page 3 — confidence: 98.1%") == 3

    def test_plain_page_heading(self):
        assert _extract_page_from_heading("Page 7") == 7

    def test_non_page_heading(self):
        assert _extract_page_from_heading("Form Fields") is None


class TestSlide:
    def test_single_chunk_when_text_short(self):
        lines = ["word " * 10]
        groups = _slide(lines, chunk_size_words=50, overlap_words=5)
        assert len(groups) == 1

    def test_multiple_chunks_when_text_long(self):
        # 10 lines × 20 words = 200 words, chunk_size=50 → multiple chunks
        lines = [("word " * 20).strip() for _ in range(10)]
        groups = _slide(lines, chunk_size_words=50, overlap_words=10)
        assert len(groups) > 1

    def test_overlap_carries_lines_forward(self):
        # 6 lines × 10 words = 60 words, chunk_size=25 → multiple chunks
        # overlap=10 → last line(s) of each chunk carry into the next
        lines = [f"line{i} " + "word " * 9 for i in range(6)]
        groups = _slide(lines, chunk_size_words=25, overlap_words=10)
        assert len(groups) > 1
        # At least one line from chunk 0 must reappear in chunk 1
        overlap_found = any(line in groups[1] for line in groups[0])
        assert overlap_found

    def test_empty_lines_returns_empty(self):
        assert _slide([], chunk_size_words=50, overlap_words=10) == []


# ------------------------------------------------------------------
# chunk_document integration
# ------------------------------------------------------------------

class TestChunkDocument:
    def test_returns_list_of_chunks(self):
        cleaned = _make_cleaned()
        chunks = chunk_document(cleaned)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_at_least_one_chunk(self):
        cleaned = _make_cleaned(page_texts=["This is a loan document with some content."])
        chunks = chunk_document(cleaned)
        assert len(chunks) >= 1

    def test_chunk_id_contains_session(self):
        cleaned = _make_cleaned(session_id="my-session")
        chunks = chunk_document(cleaned)
        assert all("my-session" in c.chunk_id for c in chunks)

    def test_chunk_index_is_sequential(self):
        cleaned = _make_cleaned(page_texts=["word " * 500])
        chunks = chunk_document(cleaned)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_form_fields_included_in_chunks(self):
        fields = [_make_field("Loan Amount", "750000")]
        cleaned = _make_cleaned(form_fields=fields)
        all_text = " ".join(c.text for c in chunk_document(cleaned))
        assert "Loan Amount" in all_text
        assert "750000" in all_text

    def test_empty_chunks_not_included(self):
        cleaned = _make_cleaned(page_texts=[""])
        chunks = chunk_document(cleaned)
        assert all(c.text.strip() != "" for c in chunks)

    def test_to_dict_is_serialisable(self):
        import json
        cleaned = _make_cleaned()
        chunks = chunk_document(cleaned)
        for c in chunks:
            json.dumps(c.to_dict())  # must not raise

    def test_word_count_matches_text(self):
        cleaned = _make_cleaned(page_texts=["hello world this is a test sentence"])
        chunks = chunk_document(cleaned)
        for c in chunks:
            assert c.word_count == len(c.text.split())

    def test_multi_page_chunks_track_page(self):
        cleaned = _make_cleaned(page_texts=[
            "Content on page one " * 20,
            "Content on page two " * 20,
        ])
        chunks = chunk_document(cleaned)
        pages_seen = {c.page for c in chunks if c.page is not None}
        assert len(pages_seen) >= 1

    def test_custom_chunk_size(self):
        # Multi-line realistic content — 20 lines × 30 words each = 600 words
        lines = ["The applicant has submitted a loan request with the following details and terms. " * 2
                 for _ in range(20)]
        long_text = "\n".join(lines)
        cleaned = _make_cleaned(page_texts=[long_text])
        chunks_small = chunk_document(cleaned, chunk_size=100, overlap=10)
        chunks_large = chunk_document(cleaned, chunk_size=400, overlap=40)
        assert len(chunks_small) > len(chunks_large)
