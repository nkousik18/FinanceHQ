"""
Tests for app/pipeline/extractor.py

Uses moto to mock AWS — no real AWS calls during tests.
Run:  pytest fastapi_service/tests/pipeline/test_extractor.py -v
"""
import json
import io
from unittest.mock import patch, MagicMock

import pytest

from app.pipeline.extractor import (
    _parse_blocks,
    _blocks_to_extraction_result,
    ExtractionError,
    ExtractionResult,
    PageResult,
)


# ------------------------------------------------------------------
# Fixtures — synthetic Textract block payloads
# ------------------------------------------------------------------

def _make_word_block(text: str, page: int, confidence: float, text_type: str = "PRINTED") -> dict:
    return {
        "BlockType": "WORD",
        "Text": text,
        "Page": page,
        "Confidence": confidence,
        "TextType": text_type,
    }


def _make_line_block(text: str, page: int) -> dict:
    return {
        "BlockType": "LINE",
        "Text": text,
        "Page": page,
        "Confidence": 99.0,
    }


SINGLE_PAGE_BLOCKS = [
    _make_line_block("Loan Agreement", page=1),
    _make_line_block("Interest Rate: 5.5%", page=1),
    _make_word_block("Loan", page=1, confidence=99.0),
    _make_word_block("Agreement", page=1, confidence=98.5),
    _make_word_block("Interest", page=1, confidence=97.0),
    _make_word_block("Rate:", page=1, confidence=96.0),
    _make_word_block("5.5%", page=1, confidence=95.0),
]

MULTI_PAGE_BLOCKS = [
    _make_line_block("Section 1", page=1),
    _make_word_block("Section", page=1, confidence=99.0),
    _make_word_block("1", page=1, confidence=98.0),
    _make_line_block("Repayment Schedule", page=2),
    _make_word_block("Repayment", page=2, confidence=91.0),
    _make_word_block("Schedule", page=2, confidence=90.0),
]

HANDWRITING_BLOCKS = [
    _make_line_block("Signature", page=1),
    _make_word_block("Signature", page=1, confidence=72.0, text_type="HANDWRITING"),
    _make_word_block("John", page=1, confidence=65.0, text_type="HANDWRITING"),
]

LOW_CONFIDENCE_BLOCKS = [
    _make_line_block("Blurry text here", page=1),
    _make_word_block("Blurry", page=1, confidence=55.0),
    _make_word_block("text", page=1, confidence=60.0),
    _make_word_block("here", page=1, confidence=58.0),
]


# ------------------------------------------------------------------
# Tests — _parse_blocks
# ------------------------------------------------------------------

class TestParseBlocks:
    def test_single_page_returns_one_page(self):
        page_map, _ = _parse_blocks(SINGLE_PAGE_BLOCKS)
        assert len(page_map) == 1
        assert 1 in page_map

    def test_page_text_joins_lines(self):
        page_map, _ = _parse_blocks(SINGLE_PAGE_BLOCKS)
        assert "Loan Agreement" in page_map[1].text
        assert "Interest Rate: 5.5%" in page_map[1].text

    def test_multi_page_returns_two_pages(self):
        page_map, _ = _parse_blocks(MULTI_PAGE_BLOCKS)
        assert len(page_map) == 2
        assert 1 in page_map
        assert 2 in page_map

    def test_confidence_calculation(self):
        page_map, _ = _parse_blocks(SINGLE_PAGE_BLOCKS)
        page = page_map[1]
        assert 90.0 < page.avg_confidence <= 100.0

    def test_handwriting_detected(self):
        page_map, _ = _parse_blocks(HANDWRITING_BLOCKS)
        assert page_map[1].has_handwriting is True

    def test_no_handwriting_when_all_printed(self):
        page_map, _ = _parse_blocks(SINGLE_PAGE_BLOCKS)
        assert page_map[1].has_handwriting is False

    def test_low_confidence_words_counted(self):
        page_map, _ = _parse_blocks(LOW_CONFIDENCE_BLOCKS)
        page = page_map[1]
        # All 3 words are below 80 confidence
        assert page.low_confidence_words == 3

    def test_empty_blocks_returns_empty_page_map(self):
        page_map, _ = _parse_blocks([])
        assert page_map == {}

    def test_blocks_without_lines_still_counts_words(self):
        # Only WORD blocks, no LINE blocks
        blocks = [_make_word_block("Hello", page=1, confidence=99.0)]
        page_map, _ = _parse_blocks(blocks)
        assert 1 in page_map
        assert page_map[1].word_count == 1


# ------------------------------------------------------------------
# Tests — _blocks_to_extraction_result
# ------------------------------------------------------------------

class TestBlocksToExtractionResult:
    def test_result_type(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS)
        assert isinstance(result, ExtractionResult)

    def test_session_id_preserved(self):
        result = _blocks_to_extraction_result("sess-abc", SINGLE_PAGE_BLOCKS)
        assert result.session_id == "sess-abc"

    def test_total_pages_count(self):
        result = _blocks_to_extraction_result("sess-001", MULTI_PAGE_BLOCKS)
        assert result.total_pages == 2

    def test_full_text_contains_all_page_content(self):
        result = _blocks_to_extraction_result("sess-001", MULTI_PAGE_BLOCKS)
        assert "Section 1" in result.full_text
        assert "Repayment Schedule" in result.full_text

    def test_page_break_separator_present(self):
        result = _blocks_to_extraction_result("sess-001", MULTI_PAGE_BLOCKS)
        assert "--- Page Break ---" in result.full_text

    def test_handwriting_flag_propagates(self):
        result = _blocks_to_extraction_result("sess-001", HANDWRITING_BLOCKS)
        assert result.has_handwriting is True

    def test_no_handwriting_flag(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS)
        assert result.has_handwriting is False

    def test_job_id_stored(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS, job_id="job-xyz")
        assert result.textract_job_id == "job-xyz"

    def test_job_id_none_for_sync(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS, job_id=None)
        assert result.textract_job_id is None

    def test_overall_confidence_is_weighted_average(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS)
        assert 0.0 < result.overall_avg_confidence <= 100.0

    def test_to_dict_is_serializable(self):
        result = _blocks_to_extraction_result("sess-001", SINGLE_PAGE_BLOCKS)
        d = result.to_dict()
        # Must be JSON serializable
        json_str = json.dumps(d)
        assert "sess-001" in json_str

    def test_empty_blocks_gives_empty_result(self):
        result = _blocks_to_extraction_result("sess-001", [])
        assert result.total_pages == 0
        assert result.full_text == ""
        assert result.overall_avg_confidence == 0.0


# ------------------------------------------------------------------
# Tests — ExtractionError raised correctly
# ------------------------------------------------------------------

class TestExtractionError:
    def test_extraction_error_is_exception(self):
        err = ExtractionError("something failed")
        assert isinstance(err, Exception)
        assert "something failed" in str(err)
