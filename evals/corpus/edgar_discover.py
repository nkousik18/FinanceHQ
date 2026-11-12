"""
R1 — Discover candidate EDGAR "deals": one credit agreement plus 2+ of its amendments.

Output (committed, for Kousik's approval): evals/corpus/edgar_candidates.json
Nothing is added to the corpus until a deal is approved (scope v2 §5.0).

How a deal is assembled:
  1. Full-text search EX-10 exhibits for amendment titles ("First Amendment to
     Credit Agreement", "Amendment No. 2 to Credit Agreement", ...), 2019–2025.
  2. Group hits by company (CIK); keep companies with 2+ amendment exhibits.
  3. Read each amendment and pull the date of the agreement it amends
     ("Credit Agreement, dated as of March 4, 2021"). Amendments that share
     that date amend the same agreement.
  4. Search the same company's EX-10 exhibits filed shortly after that date
     for the original agreement; keep the largest one whose text starts with
     "CREDIT AGREEMENT".
  5. Estimate page counts from text length; record whether a PDF copy exists.

Usage (repo root):
    venv/bin/python -m evals.corpus.edgar_discover --max-companies 40
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from evals.corpus.sec_client import SECClient

log = logging.getLogger("evals.corpus.discover")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "evals" / "corpus" / "edgar_candidates.json"

AMENDMENT_QUERIES = [
    '"First Amendment to Credit Agreement"',
    '"Second Amendment to Credit Agreement"',
    '"Third Amendment to Credit Agreement"',
    '"Amendment No. 1 to Credit Agreement"',
    '"Amendment No. 2 to Credit Agreement"',
]
CHARS_PER_PAGE = 3000          # rough: a dense legal page of rendered HTML
MIN_ORIGINAL_YEAR = 2015
MAX_ORIGINAL_PAGES = 300

_DATE = r"([A-Z][a-z]+\.?\s+\d{1,2},\s*\d{4})"
_AMENDS_RE = re.compile(
    r"Credit Agreement[,\s]*(?:dated|entered into)\s+as\s+of\s+" + _DATE, re.IGNORECASE
)
# Title of an original agreement, searched near the top (EDGAR prefixes an exhibit header).
_TITLE_RE = re.compile(r"\b(?:AMENDED AND RESTATED\s+)?(?:REVOLVING\s+|TERM\s+LOAN\s+)?CREDIT AGREEMENT\b")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: bytes) -> str:
    p = _Text()
    p.feed(raw.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def parse_date(s: str) -> Optional[datetime]:
    s = re.sub(r"\s+", " ", s.replace(".", "")).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d,%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def amended_agreement_date(text: str) -> Optional[datetime]:
    """
    Date of the agreement an amendment amends.

    Skips matches that are the amendment's own title ("Amendment No. 2 to
    Credit Agreement dated as of <amendment date>"): a match counts only if
    the 40 characters before "Credit Agreement" don't mention an amendment.
    """
    for m in _AMENDS_RE.finditer(text[:8000]):
        before = text[max(0, m.start() - 40):m.start()].lower()
        if "amendment" in before:
            continue
        return parse_date(m.group(1))
    return None


def mentions_date(text: str, d: datetime) -> bool:
    """True if `text` contains d as 'March 4, 2021' or 'March 04, 2021' (any spacing)."""
    for fmt in ("%B %-d, %Y", "%B %d, %Y"):
        pat = re.escape(d.strftime(fmt)).replace(r"\ ", r"\s*")
        if re.search(pat, text):
            return True
    return False


def _hit_record(hit: dict) -> dict:
    s = hit["_source"]
    adsh, filename = hit["_id"].split(":", 1)
    cik = s["ciks"][0]
    return {
        "cik": cik, "adsh": adsh, "filename": filename,
        "file_type": s.get("file_type", ""), "file_date": s.get("file_date", ""),
        "form": s.get("form", ""), "company": (s.get("display_names") or [""])[0],
        "sic": (s.get("sics") or [""])[0], "description": s.get("file_description") or "",
    }


def search_amendments(sec: SECClient, pages_per_query: int) -> dict[str, list[dict]]:
    by_cik: dict[str, dict[str, dict]] = defaultdict(dict)
    for q in AMENDMENT_QUERIES:
        for page in range(pages_per_query):
            res = sec.full_text_search(q, startdt="2019-01-01", enddt="2025-12-31", offset=page * 100)
            hits = res["hits"]["hits"]
            for h in hits:
                r = _hit_record(h)
                if r["file_type"].upper().startswith("EX-10") and r["filename"].lower().endswith((".htm", ".html")):
                    by_cik[r["cik"]][f'{r["adsh"]}:{r["filename"]}'] = r
            if len(hits) < 100:
                break
    log.info("amendment_search companies=%d", len(by_cik))
    return {cik: list(v.values()) for cik, v in by_cik.items() if len(v) >= 2}


def _doc_text(sec: SECClient, r: dict) -> str:
    return html_to_text(sec.get_bytes(sec.archive_url(r["cik"], r["adsh"], r["filename"])))


def _pdf_in_filing(sec: SECClient, cik: str, adsh: str) -> bool:
    try:
        idx = sec.filing_index(cik, adsh)
    except Exception:
        return False
    return any(i["name"].lower().endswith(".pdf") for i in idx.get("directory", {}).get("item", []))


def find_original(sec: SECClient, cik: str, agreement_date: datetime) -> Optional[dict]:
    res = sec.full_text_search(
        '"Credit Agreement"', ciks=cik,
        startdt=(agreement_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        enddt=(agreement_date + timedelta(days=100)).strftime("%Y-%m-%d"),
    )
    best = None
    for h in res["hits"]["hits"]:
        r = _hit_record(h)
        if not r["file_type"].upper().startswith("EX-10") or not r["filename"].lower().endswith((".htm", ".html")):
            continue
        text = _doc_text(sec, r)
        head = text[:600]
        t = _TITLE_RE.search(head)          # upper-case title, as on a cover page
        if not t or re.search(r"amendment", head[:t.end()], re.I):
            continue
        if not mentions_date(text[:5000], agreement_date):
            continue                         # a different agreement filed nearby
        r["chars"] = len(text)
        if best is None or r["chars"] > best["chars"]:
            best = r
    return best


_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth)\s+amendment\b|\bamendment\s+no\.?\s*(\d+)\b",
    re.IGNORECASE)
_ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}


def amendment_ordinal(opening: str) -> Optional[str]:
    """Amendment number from its title: 'Second Amendment' / 'Amendment No. 2' -> '2'."""
    m = _ORDINAL_RE.search(opening)
    if not m:
        return None
    return _ORDINAL_WORDS[m.group(1).lower()] if m.group(1) else m.group(2)


_OWN_DATE_RE = re.compile(r"(?:dated|entered\s+into|effective)\s+(?:as\s+of\s+)?" + _DATE, re.IGNORECASE)


def amendment_own_date(opening: str) -> Optional[datetime]:
    """The amendment's own date: the first 'dated / entered into as of <date>' in its opening."""
    m = _OWN_DATE_RE.search(opening)
    return parse_date(m.group(1)) if m else None


def dedupe_amendments(amds: list[dict]) -> Optional[list[dict]]:
    """
    Same amendment number + same own date = one document filed twice (e.g. in an
    8-K, then again in a 10-Q, or first redacted and later in full): keep the
    most complete copy: fewest "[***]" redactions, then the longest. Same number + different date = two different agreements
    share a date: the deal is ambiguous, return None. Without a readable date,
    fall back to near-identical length (within 2%).
    """
    kept: dict[str, dict] = {}
    for a in sorted(amds, key=lambda a: a["file_date"]):
        key = amendment_ordinal(a["opening"]) or a["filename"]
        if key in kept:
            k = kept[key]
            d1, d2 = amendment_own_date(k["opening"]), amendment_own_date(a["opening"])
            same = (d1 == d2) if (d1 and d2) else \
                abs(k["chars"] - a["chars"]) <= 0.02 * max(k["chars"], 1)
            if same:
                if (a.get("redactions", 0), -a["chars"]) < (k.get("redactions", 0), -k["chars"]):
                    kept[key] = a
                continue
            return None
        kept[key] = a
    return list(kept.values())


def build_deal(sec: SECClient, cik: str, amendments: list[dict]) -> Optional[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for a in amendments:
        text = _doc_text(sec, a)
        d = amended_agreement_date(text)
        if d is None:
            continue
        a["chars"] = len(text)
        a["redactions"] = text.count("[***]")
        a["opening"] = text[:800]   # long enough to pass redaction notices
        groups[d.strftime("%Y-%m-%d")].append(a)

    for date_s, amds in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        amds = dedupe_amendments(amds)
        if amds is None:
            log.info("deal_ambiguous cik=%s date=%s (two agreements share the date)", cik, date_s)
            continue
        if len(amds) < 2:
            continue
        d = datetime.strptime(date_s, "%Y-%m-%d")
        if d.year < MIN_ORIGINAL_YEAR:
            continue
        orig = find_original(sec, cik, d)
        if not orig:
            continue
        pages = round(orig["chars"] / CHARS_PER_PAGE)
        if pages > MAX_ORIGINAL_PAGES:
            continue
        amds = sorted(amds, key=lambda a: a["file_date"])
        return {
            "company": orig["company"] or amds[0]["company"],
            "cik": cik,
            "sic": orig["sic"] or amds[0]["sic"],
            "agreement_date": date_s,
            "original": {**_public(sec, orig), "est_pages": pages,
                         "has_pdf_in_filing": _pdf_in_filing(sec, cik, orig["adsh"])},
            "amendments": [{**_public(sec, a), "est_pages": max(1, round(a["chars"] / CHARS_PER_PAGE)),
                            "opening": a["opening"]} for a in amds[:3]],
        }
    return None


def _public(sec: SECClient, r: dict) -> dict:
    return {"url": sec.archive_url(r["cik"], r["adsh"], r["filename"]), "accession": r["adsh"],
            "file_type": r["file_type"], "form": r["form"], "file_date": r["file_date"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-companies", type=int, default=40)
    ap.add_argument("--pages-per-query", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sec = SECClient()
    companies = search_amendments(sec, args.pages_per_query)
    ranked = sorted(companies.items(), key=lambda kv: -len(kv[1]))[: args.max_companies]

    deals, seen_sic = [], set()
    for cik, amds in ranked:
        try:
            deal = build_deal(sec, cik, amds)
        except Exception as e:  # one bad filing shouldn't stop discovery
            log.warning("deal_failed cik=%s error=%s", cik, e)
            continue
        if deal:
            deals.append(deal)
            seen_sic.add(deal["sic"])
            log.info("deal_found %s orig_pages=%s amendments=%d",
                     deal["company"], deal["original"]["est_pages"], len(deal["amendments"]))

    out = {
        "generated_on": datetime.now().strftime("%Y-%m-%d"),
        "method": "evals/corpus/edgar_discover.py (see module docstring)",
        "sec_requests": sec.requests_made,
        "candidates": deals,
    }
    args.out.write_text(json.dumps(out, indent=2))
    log.info("wrote %s candidates=%d industries=%d requests=%d",
             args.out.relative_to(REPO), len(deals), len(seen_sic), sec.requests_made)


if __name__ == "__main__":
    main()
