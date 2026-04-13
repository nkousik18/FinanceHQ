"""
Validator for cleaned extraction output.

Checks:
  - Overall and per-page confidence thresholds
  - Required loan document fields are present
  - Handwriting pages flagged
  - Truncated values flagged
  - Minimum page count sanity check

Produces a ValidationReport — does NOT raise exceptions.
Downstream code decides whether to block or warn based on the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.pipeline.cleaner import CleanedResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    page: int | None = None
    field: str | None = None


@dataclass
class ValidationReport:
    session_id: str
    passed: bool                                       # False = extraction should not proceed to RAG
    issues: list[ValidationIssue] = field(default_factory=list)
    required_fields_found: dict[str, str] = field(default_factory=dict)   # field → value
    required_fields_missing: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    handwriting_pages: list[int] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status} | confidence={self.overall_confidence:.1f}% | "
            f"errors={len(self.errors)} warnings={len(self.warnings)} | "
            f"required_fields={len(self.required_fields_found)}/{len(self.required_fields_found) + len(self.required_fields_missing)}"
        )


# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------

CONFIDENCE_ERROR_THRESHOLD = 70.0    # below this → error, block
CONFIDENCE_WARN_THRESHOLD = 85.0     # below this → warning, proceed with caution
MIN_PAGES = 1

# Required fields for a loan application — fuzzy matched against extracted keys
# Each entry: (canonical_name, list_of_possible_key_substrings)
REQUIRED_FIELDS: list[tuple[str, list[str]]] = [
    ("applicant_first_name",    ["first name", "first_name"]),
    ("applicant_last_name",     ["last name", "last_name"]),
    ("loan_amount",             ["loan requested", "loan amount", "rupees"]),
    ("loan_tenure",             ["tenor", "tenure", "term"]),
    ("rate_type",               ["floating rate", "fixed rate", "rate of interest"]),
    ("repayment_mode",          ["clearance service", "ecs", "standing instructions", "post dated"]),
    ("gross_monthly_income",    ["gross monthly income"]),
    ("date",                    ["date:"]),
    ("branch",                  ["branch"]),
    ("applicant_pan",           ["pan card"]),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fuzzy_match(key: str, substrings: list[str]) -> bool:
    key_lower = key.lower()
    return any(s in key_lower for s in substrings)


def _find_required_fields(
    cleaned: CleanedResult,
) -> tuple[dict[str, str], list[str]]:
    """
    Match required fields against extracted form fields.
    Returns (found_dict, missing_list).
    """
    found: dict[str, str] = {}
    missing: list[str] = []

    for canonical, substrings in REQUIRED_FIELDS:
        match = None
        for f in cleaned.form_fields:
            if _fuzzy_match(f.key, substrings) and f.value:
                match = f
                break
        if match:
            found[canonical] = match.value
        else:
            missing.append(canonical)

    return found, missing


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def validate_extraction(cleaned: CleanedResult) -> ValidationReport:
    """
    Run all validation checks on a CleanedResult.
    Returns a ValidationReport — never raises.
    """
    logger.info("validation_start", session_id=cleaned.session_id)

    issues: list[ValidationIssue] = []

    # 1. Page count sanity
    if cleaned.total_pages < MIN_PAGES:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            code="NO_PAGES",
            message=f"Extraction returned 0 pages. PDF may be empty or corrupt.",
        ))

    # 2. Overall confidence
    conf = cleaned.overall_avg_confidence
    if conf < CONFIDENCE_ERROR_THRESHOLD:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            code="LOW_OVERALL_CONFIDENCE",
            message=f"Overall confidence {conf:.1f}% is below error threshold {CONFIDENCE_ERROR_THRESHOLD}%. "
                    f"Document may be too blurry or low quality for reliable extraction.",
        ))
    elif conf < CONFIDENCE_WARN_THRESHOLD:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="MODERATE_OVERALL_CONFIDENCE",
            message=f"Overall confidence {conf:.1f}% is below warning threshold {CONFIDENCE_WARN_THRESHOLD}%. "
                    f"Some extracted text may be inaccurate.",
        ))

    # 3. Per-page confidence
    for page in cleaned.pages:
        if page.avg_confidence < CONFIDENCE_ERROR_THRESHOLD:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                code="LOW_PAGE_CONFIDENCE",
                message=f"Page {page.page_number} confidence {page.avg_confidence:.1f}% below threshold.",
                page=page.page_number,
            ))
        elif page.avg_confidence < CONFIDENCE_WARN_THRESHOLD:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="MODERATE_PAGE_CONFIDENCE",
                message=f"Page {page.page_number} confidence {page.avg_confidence:.1f}% is moderate.",
                page=page.page_number,
            ))

    # 4. Handwriting pages
    hw_pages = [p.page_number for p in cleaned.pages if p.has_handwriting]
    if hw_pages:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            code="HANDWRITING_DETECTED",
            message=f"Handwriting detected on page(s): {hw_pages}. "
                    f"Handwritten values may have lower accuracy.",
        ))

    # 5. Truncated fields
    for f in cleaned.form_fields:
        if f.is_truncated:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="TRUNCATED_VALUE",
                message=f"Field '{f.key}' value '{f.value}' appears truncated.",
                page=f.page,
                field=f.key,
            ))

    # 6. Required fields
    found, missing = _find_required_fields(cleaned)
    if missing:
        for m in missing:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="MISSING_REQUIRED_FIELD",
                message=f"Required field '{m}' not found or has no value.",
                field=m,
            ))

    # 7. Empty extraction (no form fields AND no table data)
    if not cleaned.form_fields and not cleaned.tables:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            code="NO_STRUCTURED_DATA",
            message="No form fields or tables extracted. Document may not be a structured loan form.",
        ))

    # Pass = no ERROR-level issues
    passed = not any(i.severity == Severity.ERROR for i in issues)

    report = ValidationReport(
        session_id=cleaned.session_id,
        passed=passed,
        issues=issues,
        required_fields_found=found,
        required_fields_missing=missing,
        overall_confidence=conf,
        handwriting_pages=hw_pages,
    )

    logger.info(
        "validation_complete",
        session_id=cleaned.session_id,
        passed=passed,
        errors=len(report.errors),
        warnings=len(report.warnings),
        required_found=len(found),
        required_missing=len(missing),
        summary=report.summary(),
    )

    return report
