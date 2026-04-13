"""
Generates a clean Markdown file from cleaned extraction + validation results.

Output structure:
  # Document Title
  ## Extraction Summary
  ## Validation Report
  ## Form Fields
  ## Tables
  ## Full Text (page by page)

This markdown file is what gets saved to S3 and used as the primary
input for chunking → embedding → RAG.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.pipeline.cleaner import CleanedResult
from app.pipeline.validator import ValidationReport, Severity
from app.core.logging import get_logger

logger = get_logger(__name__)


def _severity_icon(severity: Severity) -> str:
    return {"info": "ℹ", "warning": "⚠", "error": "✗"}.get(severity.value, "•")


def generate_markdown(
    cleaned: CleanedResult,
    report: ValidationReport,
) -> str:
    """
    Build the full markdown document string from cleaned data + validation report.
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    applicant_name = (
        cleaned.form_fields
        and next(
            (f.value for f in cleaned.form_fields if "first name" in f.key.lower()),
            None,
        )
    )
    title = f"Loan Application — {applicant_name}" if applicant_name else "Loan Application"

    lines += [
        f"# {title}",
        "",
        f"**Session:** `{cleaned.session_id}`  ",
        f"**Extracted:** {now}  ",
        f"**Pages:** {cleaned.total_pages}  ",
        f"**Overall Confidence:** {cleaned.overall_avg_confidence:.1f}%  ",
        f"**Handwriting Detected:** {'Yes — pages ' + str(report.handwriting_pages) if cleaned.has_handwriting else 'No'}  ",
        f"**Validation:** {'PASS' if report.passed else 'FAIL'}",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Cleaning notes
    # ------------------------------------------------------------------
    if cleaned.cleaning_notes:
        lines += ["## Cleaning Notes", ""]
        for note in cleaned.cleaning_notes:
            lines.append(f"- {note}")
        lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------
    lines += ["## Validation Report", ""]

    if not report.issues:
        lines.append("No issues found.")
    else:
        for issue in report.issues:
            icon = _severity_icon(issue.severity)
            location = ""
            if issue.page:
                location += f" (page {issue.page})"
            if issue.field:
                location += f" [field: {issue.field}]"
            lines.append(f"- {icon} **{issue.code}**{location}: {issue.message}")

    lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # Required fields summary
    # ------------------------------------------------------------------
    lines += ["## Key Loan Fields", ""]
    lines += ["| Field | Value |", "|---|---|"]

    for canonical, value in sorted(report.required_fields_found.items()):
        display_key = canonical.replace("_", " ").title()
        lines.append(f"| {display_key} | {value} |")

    for missing in report.required_fields_missing:
        display_key = missing.replace("_", " ").title()
        lines.append(f"| {display_key} | ⚠ Not found |")

    lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # All form fields
    # ------------------------------------------------------------------
    lines += ["## All Form Fields", ""]
    lines += ["| Page | Field | Value | Notes |", "|---|---|---|---|"]

    for f in cleaned.form_fields:
        value = f.value if f.value else "_empty_"
        notes = []
        if f.is_truncated:
            notes.append("⚠ truncated")
        if f.original_value is not None:
            notes.append(f"normalised from `{f.original_value}`")
        note_str = ", ".join(notes)
        lines.append(f"| {f.page} | {f.key} | {value} | {note_str} |")

    lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    if cleaned.tables:
        lines += ["## Tables", ""]
        for i, table in enumerate(cleaned.tables, 1):
            lines += [f"### Table {i} — Page {table.page} ({table.rows} rows × {table.cols} cols)", ""]
            md_table = table.to_markdown()
            if md_table:
                lines.append(md_table)
            else:
                lines.append("_No cell content extracted._")
            lines += [""]
        lines += ["---", ""]

    # ------------------------------------------------------------------
    # Full text — page by page
    # ------------------------------------------------------------------
    lines += ["## Full Text", ""]

    for page in cleaned.pages:
        hw_note = " _(contains handwriting)_" if page.has_handwriting else ""
        conf_note = f" — confidence: {page.avg_confidence:.1f}%"
        lines += [
            f"### Page {page.page_number}{hw_note}{conf_note}",
            "",
            page.text,
            "",
        ]

    return "\n".join(lines)


def write_markdown(
    cleaned: CleanedResult,
    report: ValidationReport,
) -> str:
    """
    Generate and return the markdown string.
    Caller is responsible for saving to S3.
    """
    logger.info("markdown_generation_start", session_id=cleaned.session_id)
    md = generate_markdown(cleaned, report)
    logger.info(
        "markdown_generation_complete",
        session_id=cleaned.session_id,
        characters=len(md),
        lines=md.count("\n"),
    )
    return md
