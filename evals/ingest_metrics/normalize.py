"""
Text and number normalization for the evals (scope v2 D10).

Rules (recorded in the Decision Log):
- text:  Unicode NFKC; curly quotes -> straight; en/em/minus dashes -> "-";
         non-breaking and other spaces -> " "; whitespace collapsed; ends stripped.
         Case is kept: CER measures exact text, and casing is real content.
- match: text rules + lower case + punctuation other than . , % $ / - removed.
         Used for cell, key-value and header/footer matching, where layout
         punctuation (colons, bullets) shouldn't decide a match.
- numbers: amounts, rates, dates and plain numbers are pulled out as tokens.
         "$1,234.50" -> "1234.50", "3.875%" -> "3.875%", "(1,000)" -> "-1000",
         "03/15/2029" -> "03/15/2029", "2.00 percent" -> "2.00%".
         Thousands separators of any grouping (incl. Indian 7,50,000) are removed.
         Section numbers like "2.11(b)" and "Section 1.01" are kept: they are numeric
         content a reader must get right.
"""
from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                         "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                         "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                         "―": "-", "−": "-", " ": " ", " ": " ", " ": " "})
_WS = re.compile(r"\s+")
_MATCH_PUNCT = re.compile(r"[^\w\s.,%$/-]")


def norm_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return _WS.sub(" ", s).strip()


def norm_match(s: str | None) -> str:
    s = _MATCH_PUNCT.sub(" ", norm_text(s).lower())
    return _WS.sub(" ", s).strip()


# A number: optional ( - $ ; digits with any comma grouping; optional decimals; optional % / percent; optional )
_NUM = re.compile(
    r"(?P<neg>\()?(?P<sign>(?<![\w.])-)?(?:\$\s?)?"
    r"(?P<int>\d{1,3}(?:,\d{2,3})+|\d+)"
    r"(?P<dec>\.\d+)?"
    r"(?P<pct>\s?%|\s?percent\b)?"
    r"(?P<close>\))?",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def numeric_tokens(s: str | None) -> list[str]:
    """All numeric tokens in reading order, normalized (see module docstring)."""
    s = norm_text(s)
    out: list[str] = []
    dates = [(m.start(), m.end(), m.group(0)) for m in _DATE.finditer(s)]
    taken = [(a, b) for a, b, _ in dates]
    found = [(a, d) for a, _, d in dates]
    for m in _NUM.finditer(s):
        if any(a <= m.start() < b for a, b in taken):
            continue
        # a digit glued to letters (e.g. "H24B", "10b5") is an identifier, not a number
        if (m.start() > 0 and s[m.start() - 1].isalpha()) or (m.end() < len(s) and s[m.end()].isalpha()
                                                               and not m.group("pct")):
            continue
        val = m.group("int").replace(",", "") + (m.group("dec") or "")
        if m.group("pct"):
            val += "%"
        if m.group("sign") or (m.group("neg") and m.group("close")):
            val = "-" + val
        found.append((m.start(), val))
    return [v for _, v in sorted(found)]
