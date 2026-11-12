"""
Corpus manifest (scope v2 §5.0): one row per document, committed at
evals/corpus/manifest.csv. The fetch script re-creates every PDF from it.

Checksums are of the downloaded SOURCE file (EDGAR HTML), which is stable.
Rendered PDFs are not byte-stable (Chrome embeds timestamps), so they are
identified by page count instead.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from evals.corpus.edgar_discover import amendment_ordinal

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "evals" / "corpus" / "manifest.csv"
SELECTION = REPO / "evals" / "corpus" / "edgar_selection.json"
CANDIDATES = REPO / "evals" / "corpus" / "edgar_candidates.json"


@dataclass
class DocRow:
    doc_id: str
    part: str                 # edgar | consumer | simulated_scan
    deal_id: str = ""         # session for EDGAR docs: agreement + amendments
    role: str = ""            # original | amendment_<n> | form
    company: str = ""
    cik: str = ""
    accession: str = ""
    filing_date: str = ""
    form: str = ""
    file_type: str = ""
    source_url: str = ""
    source_format: str = ""   # pdf | html_rendered | image_pdf
    source_sha256: str = ""   # filled by `fetch --freeze`
    pdf_pages: str = ""       # filled by `fetch --freeze`
    renderer: str = ""        # e.g. "chrome 156.0.8078.80"
    render_zoom: str = ""     # zoom that kept original pages intact (see render.py)
    source_pages: str = ""    # page breaks in the source + 1


FIELDS = [f.name for f in fields(DocRow)]


def read_manifest(path: Path = MANIFEST) -> list[DocRow]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [DocRow(**{k: row.get(k, "") for k in FIELDS}) for row in csv.DictReader(fh)]


def write_manifest(rows: list[DocRow], path: Path = MANIFEST) -> None:
    rows = sorted(rows, key=lambda r: (r.part, r.doc_id))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def edgar_rows(selection: dict, candidates: dict) -> list[DocRow]:
    """Build EDGAR rows for the approved deals, in selection order."""
    by_cik = {c["cik"]: c for c in candidates["candidates"]}
    rows: list[DocRow] = []
    for deal in selection["deals"]:
        c = by_cik.get(deal["cik"])
        if c is None:
            raise KeyError(f'{deal["deal_id"]}: CIK {deal["cik"]} not in candidates file')
        did = deal["deal_id"]

        def row(doc: dict, doc_id: str, role: str) -> DocRow:
            return DocRow(
                doc_id=doc_id, part="edgar", deal_id=did, role=role, company=deal["name"],
                cik=c["cik"], accession=doc["accession"], filing_date=doc["file_date"],
                form=doc["form"], file_type=doc["file_type"], source_url=doc["url"],
                source_format="html_rendered",
            )

        rows.append(row(c["original"], f"{did}_ca", "original"))
        used: set[str] = set()
        for i, a in enumerate(c["amendments"], start=1):
            n = amendment_ordinal(a.get("opening", "")) or str(i)
            if n in used:
                n = f"{n}b"
            used.add(n)
            rows.append(row(a, f"{did}_am{n}", f"amendment_{n}"))
    return rows


def build_manifest(path: Path = MANIFEST) -> list[DocRow]:
    """
    Rebuild the EDGAR and consumer rows (from the approved selection and the
    CFPB form list), keeping frozen values for unchanged documents and any
    other parts (e.g. simulated scans) as they are.
    """
    from evals.corpus.cfpb import consumer_rows

    selection = json.loads(SELECTION.read_text())
    candidates = json.loads(CANDIDATES.read_text())
    existing = {r.doc_id: r for r in read_manifest(path)}
    new = edgar_rows(selection, candidates) + consumer_rows()
    for r in new:
        old = existing.get(r.doc_id)
        if old and old.source_url == r.source_url:
            for f in ("source_sha256", "pdf_pages", "renderer", "render_zoom", "source_pages"):
                setattr(r, f, getattr(old, f))
    others = [r for r in existing.values() if r.part not in ("edgar", "consumer")]
    rows = others + new
    write_manifest(rows, path)
    return rows
