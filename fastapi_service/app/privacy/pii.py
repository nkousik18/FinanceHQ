"""
PII detection and redaction — STANDALONE PLACEHOLDER (R0).

Not wired into the ingestion pipeline, the query API, or the evals. Nothing
imports this module except its tests and the placeholder Airflow DAG. It
exists so the PII stage has a defined interface before it is connected.

Pattern style follows LoanGuardrails.PII_PATTERNS in the sibling project
loan-doc-ai-end-to-end, with two changes:
  - checksum validation (Luhn for cards, ABA for routing numbers) to cut
    false positives on loan figures, which are full of long digit runs
  - each match carries its span, so redaction is offset-safe and auditable

The evals corpus is public (SEC filings, CFPB sample forms), so it contains
company data, not personal data. This module matters for user-uploaded
documents in production, once it is connected.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class PIIType(str, Enum):
    SSN = "ssn"
    EIN = "ein"
    CREDIT_CARD = "credit_card"
    ROUTING_NUMBER = "routing_number"
    BANK_ACCOUNT = "bank_account"
    EMAIL = "email"
    PHONE = "phone"
    PAN = "pan"            # Indian Permanent Account Number (legacy loan1.pdf is Indian)


class RedactionMode(str, Enum):
    MASK = "mask"          # "[SSN]"
    HASH = "hash"          # "[SSN:3f2a9c1e]" — same value → same tag, for joins/audit
    PARTIAL = "partial"    # keep last 4 characters: "*****6789"


@dataclass(frozen=True)
class PIIMatch:
    type: PIIType
    start: int
    end: int
    value: str


@dataclass
class RedactionResult:
    text: str
    matches: list[PIIMatch] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.type.value] = out.get(m.type.value, 0) + 1
        return out


# ------------------------------------------------------------------
# Checksums
# ------------------------------------------------------------------

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    d = _digits(number)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def aba_valid(number: str) -> bool:
    d = _digits(number)
    if len(d) != 9:
        return False
    w = [3, 7, 1] * 3
    return sum(int(c) * k for c, k in zip(d, w)) % 10 == 0


# ------------------------------------------------------------------
# Patterns — ordered: more specific types first, so overlaps resolve to them
# ------------------------------------------------------------------

_PATTERNS: list[tuple[PIIType, re.Pattern, Optional[Callable[[str], bool]]]] = [
    (PIIType.EMAIL,
     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    (PIIType.CREDIT_CARD,
     re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), luhn_valid),
    (PIIType.SSN,
     re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    (PIIType.EIN,
     re.compile(r"\b\d{2}-\d{7}\b"), None),
    (PIIType.PAN,
     re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    (PIIType.PHONE,
     re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"), None),
    (PIIType.ROUTING_NUMBER,
     # Only when labelled: a bare valid-checksum 9-digit number is too common.
     re.compile(r"(?i)(?:routing|aba|rtn)\D{0,20}(\d{9})\b"), aba_valid),
    (PIIType.BANK_ACCOUNT,
     # Only when labelled, to avoid flagging loan amounts and reference numbers.
     re.compile(r"(?i)(?:account|acct|a/c)\s*(?:no\.?|number|#)?\s*[:\-]?\s*(\d{6,17})\b"), None),
]


def detect(text: str) -> list[PIIMatch]:
    """Return non-overlapping PII matches, sorted by position."""
    found: list[PIIMatch] = []
    taken: list[tuple[int, int]] = []

    for pii_type, pattern, validator in _PATTERNS:
        for m in pattern.finditer(text):
            # Labelled patterns capture the value in group 1; redact only that.
            grp = 1 if m.groups() else 0
            start, end, value = m.start(grp), m.end(grp), m.group(grp)
            if validator and not validator(value):
                continue
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            found.append(PIIMatch(pii_type, start, end, value))

    found.sort(key=lambda x: x.start)
    return found


def _replacement(match: PIIMatch, mode: RedactionMode, salt: str) -> str:
    tag = match.type.value.upper()
    if mode is RedactionMode.MASK:
        return f"[{tag}]"
    if mode is RedactionMode.HASH:
        digest = hashlib.sha256((salt + match.value).encode()).hexdigest()[:8]
        return f"[{tag}:{digest}]"
    keep = match.value[-4:]
    return "*" * max(0, len(match.value) - 4) + keep


def redact(text: str, mode: RedactionMode = RedactionMode.MASK, salt: str = "") -> RedactionResult:
    """
    Replace detected PII in `text`.

    `salt` only matters for HASH mode; set it from an environment variable in
    production so tags can't be reversed by hashing guessed values.
    """
    matches = detect(text)
    out, cursor = [], 0
    for m in matches:
        out.append(text[cursor:m.start])
        out.append(_replacement(m, mode, salt))
        cursor = m.end
    out.append(text[cursor:])

    result = RedactionResult(text="".join(out), matches=matches)
    logger.info("pii_redacted", mode=mode.value, total=len(matches), **result.counts)
    return result
