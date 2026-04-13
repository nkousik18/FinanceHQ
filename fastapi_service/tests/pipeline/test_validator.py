"""
Tests for app/pipeline/validator.py
Run: pytest fastapi_service/tests/pipeline/test_validator.py -v
"""
import pytest
from app.pipeline.cleaner import CleanedResult, CleanedPage, CleanedFormField, CleanedTable
from app.pipeline.validator import (
    validate_extraction, ValidationReport, Severity,
    CONFIDENCE_ERROR_THRESHOLD, CONFIDENCE_WARN_THRESHOLD,
)


def _make_cleaned(
    session_id: str = "sess-001",
    pages: list[CleanedPage] | None = None,
    form_fields: list[CleanedFormField] | None = None,
    tables: list[CleanedTable] | None = None,
    overall_conf: float = 98.0,
    has_handwriting: bool = False,
) -> CleanedResult:
    default_pages = pages or [CleanedPage(page_number=1, text="Loan text", avg_confidence=98.0, has_handwriting=False)]
    return CleanedResult(
        session_id=session_id,
        pages=default_pages,
        form_fields=form_fields or [],
        tables=tables or [],
        total_pages=len(default_pages),
        overall_avg_confidence=overall_conf,
        has_handwriting=has_handwriting,
    )


def _make_field(key: str, value: str, page: int = 1, truncated: bool = False) -> CleanedFormField:
    return CleanedFormField(key=key, value=value, page=page, confidence=95.0, is_truncated=truncated)


def _make_page(num: int = 1, conf: float = 98.0, hw: bool = False) -> CleanedPage:
    return CleanedPage(page_number=num, text="text", avg_confidence=conf, has_handwriting=hw)


class TestValidateExtraction:

    def test_returns_validation_report(self):
        cleaned = _make_cleaned()
        report = validate_extraction(cleaned)
        assert isinstance(report, ValidationReport)

    def test_pass_on_good_extraction(self):
        fields = [
            _make_field("First Name", "ARJUN"),
            _make_field("Last Name", "IYER"),
            _make_field("Loan requested from Bank- Rupees", "7,50,000", page=6),
            _make_field("Maximum Tenor/Term", "36 MONTHS", page=6),
            _make_field("Fixed Rate / Floating Rate", "FLOATING", page=6),
            _make_field("2. Electronic Clearance Service (ECS)", "ECS", page=6),
            _make_field("Gross Monthly Income", "120000", page=5),
            _make_field("Date:", "01/12/2015", page=7),
            _make_field("Branch", "INDIRANAGAR", page=7),
            _make_field("PAN Card", "AABP14589M", page=3),
        ]
        cleaned = _make_cleaned(form_fields=fields)
        report = validate_extraction(cleaned)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_fail_on_zero_pages(self):
        cleaned = _make_cleaned(pages=[])
        cleaned.total_pages = 0
        report = validate_extraction(cleaned)
        assert report.passed is False
        codes = [i.code for i in report.issues]
        assert "NO_PAGES" in codes

    def test_error_on_very_low_confidence(self):
        cleaned = _make_cleaned(overall_conf=60.0)
        report = validate_extraction(cleaned)
        assert report.passed is False
        codes = [i.code for i in report.issues]
        assert "LOW_OVERALL_CONFIDENCE" in codes

    def test_warning_on_moderate_confidence(self):
        cleaned = _make_cleaned(overall_conf=80.0)
        report = validate_extraction(cleaned)
        assert report.passed is True   # warning, not error
        codes = [i.code for i in report.issues]
        assert "MODERATE_OVERALL_CONFIDENCE" in codes

    def test_handwriting_info_issue_added(self):
        pages = [_make_page(num=1, hw=True)]
        cleaned = _make_cleaned(pages=pages, has_handwriting=True)
        report = validate_extraction(cleaned)
        codes = [i.code for i in report.issues]
        assert "HANDWRITING_DETECTED" in codes
        assert 1 in report.handwriting_pages

    def test_truncated_field_adds_warning(self):
        fields = [_make_field("Date", "15/02/200", truncated=True)]
        cleaned = _make_cleaned(form_fields=fields)
        report = validate_extraction(cleaned)
        codes = [i.code for i in report.issues]
        assert "TRUNCATED_VALUE" in codes

    def test_no_structured_data_is_error(self):
        cleaned = _make_cleaned(form_fields=[], tables=[])
        report = validate_extraction(cleaned)
        assert report.passed is False
        codes = [i.code for i in report.issues]
        assert "NO_STRUCTURED_DATA" in codes

    def test_missing_required_field_is_warning(self):
        cleaned = _make_cleaned(form_fields=[])
        report = validate_extraction(cleaned)
        assert "applicant_first_name" in report.required_fields_missing

    def test_summary_string_contains_pass_fail(self):
        cleaned = _make_cleaned()
        report = validate_extraction(cleaned)
        assert "PASS" in report.summary() or "FAIL" in report.summary()

    def test_per_page_low_confidence_error(self):
        pages = [_make_page(num=1, conf=65.0)]
        cleaned = _make_cleaned(pages=pages, overall_conf=65.0)
        report = validate_extraction(cleaned)
        codes = [i.code for i in report.issues]
        assert "LOW_PAGE_CONFIDENCE" in codes
