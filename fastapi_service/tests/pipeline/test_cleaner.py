"""
Tests for app/pipeline/cleaner.py
Run: pytest fastapi_service/tests/pipeline/test_cleaner.py -v
"""
import pytest
from app.pipeline.cleaner import (
    _clean_number,
    _clean_digit_letter_concat,
    _is_truncated,
    _clean_value,
    _is_header_noise,
    clean_extraction,
    CleanedResult,
)
from app.pipeline.extractor import (
    ExtractionResult, PageResult, FormField, Table, TableCell
)


# ------------------------------------------------------------------
# _clean_number
# ------------------------------------------------------------------

class TestCleanNumber:
    def test_spaces_between_digits_removed(self):
        assert _clean_number("110 000") == "110000"

    def test_triple_space_number(self):
        assert _clean_number("55 00 000") == "5500000"

    def test_trailing_period_removed(self):
        assert _clean_number("15 000.") == "15000"

    def test_indian_comma_format_preserved(self):
        assert _clean_number("7,50,000") == "7,50,000"

    def test_plain_number_unchanged(self):
        assert _clean_number("120000") == "120000"

    def test_non_numeric_string_unchanged(self):
        assert _clean_number("ARJUN RAMESH") == "ARJUN RAMESH"

    def test_single_number_unchanged(self):
        assert _clean_number("36") == "36"

    def test_empty_string(self):
        assert _clean_number("") == ""


# ------------------------------------------------------------------
# _clean_digit_letter_concat
# ------------------------------------------------------------------

class TestCleanDigitLetterConcat:
    def test_months_separated(self):
        assert _clean_digit_letter_concat("36MONTHS") == "36 MONTHS"

    def test_lakh_separated(self):
        assert _clean_digit_letter_concat("3200PaLAKH") == "3200 PaLAKH"

    def test_no_change_when_already_spaced(self):
        assert _clean_digit_letter_concat("36 MONTHS") == "36 MONTHS"

    def test_plain_text_unchanged(self):
        assert _clean_digit_letter_concat("FLOATING") == "FLOATING"

    def test_plain_number_unchanged(self):
        assert _clean_digit_letter_concat("120000") == "120000"


# ------------------------------------------------------------------
# _is_truncated
# ------------------------------------------------------------------

class TestIsTruncated:
    def test_truncated_date_detected(self):
        assert _is_truncated("15/02/200") is True

    def test_truncated_url_detected(self):
        assert _is_truncated("https://in.shinanglobal.c") is True

    def test_valid_date_not_truncated(self):
        assert _is_truncated("15/02/2009") is False

    def test_number_not_truncated(self):
        assert _is_truncated("750000") is False

    def test_full_word_not_truncated(self):
        assert _is_truncated("FLOATING") is False


# ------------------------------------------------------------------
# _is_header_noise
# ------------------------------------------------------------------

class TestIsHeaderNoise:
    def test_shinhan_bank_is_noise(self):
        assert _is_header_noise("Shinhan Bank") is True

    def test_single_s_is_noise(self):
        assert _is_header_noise("S") is True

    def test_india_is_noise(self):
        assert _is_header_noise("India") is True

    def test_loan_content_not_noise(self):
        assert _is_header_noise("Loan Amount: 7,50,000") is False

    def test_applicant_name_not_noise(self):
        assert _is_header_noise("ARJUN RAMESH IYER") is False


# ------------------------------------------------------------------
# clean_extraction — integration
# ------------------------------------------------------------------

def _make_extraction(
    session_id: str = "sess-001",
    pages: list[PageResult] | None = None,
    form_fields: list[FormField] | None = None,
    tables: list[Table] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        session_id=session_id,
        full_text="",
        pages=pages or [],
        form_fields=form_fields or [],
        tables=tables or [],
        total_pages=len(pages or []),
        overall_avg_confidence=98.0,
        has_handwriting=False,
    )


def _make_page(page_number: int = 1, text: str = "Some text", has_hw: bool = False) -> PageResult:
    return PageResult(
        page_number=page_number,
        text=text,
        word_count=10,
        avg_confidence=98.0,
        low_confidence_words=0,
        has_handwriting=has_hw,
    )


def _make_field(key: str, value: str, page: int = 1, conf: float = 95.0) -> FormField:
    return FormField(key=key, value=value, key_confidence=conf, value_confidence=conf, page=page)


class TestCleanExtraction:
    def test_returns_cleaned_result(self):
        ex = _make_extraction(pages=[_make_page()])
        result = clean_extraction(ex)
        assert isinstance(result, CleanedResult)

    def test_session_id_preserved(self):
        ex = _make_extraction(session_id="abc-123", pages=[_make_page()])
        result = clean_extraction(ex)
        assert result.session_id == "abc-123"

    def test_header_noise_removed_from_page(self):
        page = _make_page(text="Shinhan Bank\nLoan Amount: 7,50,000\nS\nIndia")
        ex = _make_extraction(pages=[page])
        result = clean_extraction(ex)
        assert "Shinhan Bank" not in result.pages[0].text
        assert "Loan Amount" in result.pages[0].text

    def test_empty_checkbox_fields_removed(self):
        fields = [
            _make_field("Male", ""),
            _make_field("Female", ""),
            _make_field("Loan Amount", "750000"),
        ]
        ex = _make_extraction(pages=[_make_page()], form_fields=fields)
        result = clean_extraction(ex)
        keys = [f.key for f in result.form_fields]
        assert "Male" not in keys
        assert "Female" not in keys
        assert "Loan Amount" in keys

    def test_number_values_normalised(self):
        fields = [_make_field("Net Income", "110 000")]
        ex = _make_extraction(pages=[_make_page()], form_fields=fields)
        result = clean_extraction(ex)
        assert result.form_fields[0].value == "110000"
        assert result.form_fields[0].original_value == "110 000"

    def test_duplicate_fields_deduplicated_keep_non_empty(self):
        fields = [
            _make_field("Branch", "", page=2),
            _make_field("Branch", "INDIRANAGAR", page=2),
        ]
        ex = _make_extraction(pages=[_make_page()], form_fields=fields)
        result = clean_extraction(ex)
        branch_fields = [f for f in result.form_fields if f.key == "Branch"]
        assert len(branch_fields) == 1
        assert branch_fields[0].value == "INDIRANAGAR"

    def test_truncated_field_flagged(self):
        fields = [_make_field("Date", "15/02/200")]
        ex = _make_extraction(pages=[_make_page()], form_fields=fields)
        result = clean_extraction(ex)
        assert result.form_fields[0].is_truncated is True

    def test_cleaning_notes_populated(self):
        page = _make_page(text="Shinhan Bank\nLoan text")
        fields = [_make_field("Net Income", "110 000")]
        ex = _make_extraction(pages=[page], form_fields=fields)
        result = clean_extraction(ex)
        assert len(result.cleaning_notes) > 0
