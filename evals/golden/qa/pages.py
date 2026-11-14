"""
R3 drafting, step 1 of 3: choose the pages the LLM drafts questions from (scope v2 §5.3).

No LLM here. Each **job** = one LLM call: a question type, a session, the page(s) shown to the
model, and how many candidates to ask for. Single-page jobs come from this module; cross_doc
jobs (amendment section + the original section it changes, LE + CD pairs) from `pairs.py`.

Page text = the text the validator and chunkers use: parse downstream_config("P2", part),
cleaner v2 (`evals.golden.qa.validate.CacheText`). So a span copied from the page shown to the
model is a verbatim span of the validated text.

Plan (≈ 1.5 candidates per final question, Kousik 2026-11-13; final quotas in §5.1):
- buckets: `edgar` (EDGAR digital pages), `consumer` (CFPB digital pages), `scan` (simulated
  scans, their Apple Vision text). Scan questions fill the "≥ 8 with scan evidence" quota;
  consumer + consumer scans the "≥ 20% consumer" quota.
- per (type, bucket) a number of jobs (`PLAN`), 2 candidates each.
- page rules per type: `table` needs a table (≥ 3 rows) on the page; `calculation` a table or
  ≥ 6 numbers; `explain` EDGAR prose (≥ 200 words, no table); everything else ≥ 80 words.
  Never: CFPB cover sheets, signature pages, table-of-contents pages.
- spread: sessions in seeded round robin; within a session a seeded random eligible page;
  no page is used twice (a scan also uses up its source page).
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from evals.corpus.cfpb import is_cover_page
from evals.corpus.scans import is_signature_page
from evals.ingest_metrics.normalize import numeric_tokens

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "results" / "r3"
PAGES_CACHE = OUT_DIR / "page_features.json"

SEED = 20261113
CANDIDATES_PER_JOB = 2
MIN_WORDS = 80
MIN_EXPLAIN_WORDS = 200
MIN_CALC_NUMBERS = 6
MIN_TABLE_ROWS = 3

# (type, bucket) -> jobs. 80 single-page jobs = 160 candidates; pairs.py adds 12 cross_doc jobs.
PLAN: dict[tuple[str, str], int] = {
    ("lookup", "edgar"): 13, ("lookup", "consumer"): 4, ("lookup", "scan"): 2,
    ("table", "edgar"): 10, ("table", "consumer"): 6, ("table", "scan"): 3,
    ("calculation", "edgar"): 9, ("calculation", "consumer"): 5, ("calculation", "scan"): 1,
    ("explain", "edgar"): 10, ("explain", "scan"): 2,
    ("unanswerable", "edgar"): 11, ("unanswerable", "consumer"): 4,
}

_TOC_LINE = re.compile(r"(?:\.{4,}|…{2,})\s*\d*\s*$|\s\d{1,3}\s*$")


@dataclass(frozen=True)
class PageInfo:
    doc_id: str
    part: str            # edgar | consumer | simulated_scan
    session: str         # manifest deal_id
    role: str
    page: int
    words: int
    numbers: int         # numeric tokens on the page
    table_rows: int      # rows of the largest table on the page (0: none)
    cover: bool = False
    signature: bool = False
    toc: bool = False
    scan_of: tuple[str, int] | None = None   # simulated scans: (source doc, source page)

    @property
    def bucket(self) -> str:
        return {"edgar": "edgar", "consumer": "consumer", "simulated_scan": "scan"}[self.part]

    @property
    def usable(self) -> bool:
        return self.words >= MIN_WORDS and not (self.cover or self.signature or self.toc)


@dataclass(frozen=True)
class Job:
    job_id: str
    qtype: str
    bucket: str
    session_id: str
    pages: tuple[tuple[str, int], ...]     # (doc_id, page) shown to the model, in order
    n: int = CANDIDATES_PER_JOB
    note: str = ""                         # pairs.py: what the pair is (e.g. "Section 2.11")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pages"] = [list(p) for p in self.pages]
        return d


def is_toc(text: str) -> bool:
    """Table-of-contents page: most non-blank lines end in dot leaders or a bare page number."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    return sum(bool(_TOC_LINE.search(ln)) for ln in lines) / len(lines) >= 0.5


def parse_scan_of(role: str) -> tuple[str, int] | None:
    """'scan_of:cfpb_cd_h25b:p2' -> ('cfpb_cd_h25b', 2)."""
    if not role.startswith("scan_of:"):
        return None
    _, doc, p = role.split(":")
    return doc, int(p.lstrip("p"))


def page_infos(src, part: str, session: str, role: str) -> list[PageInfo]:
    """Features of every page of one SourceDoc (evals.chunking.base.SourceDoc)."""
    rows_on: dict[int, int] = defaultdict(int)
    for t in src.tables:
        rows_on[t.page] = max(rows_on[t.page], sum(any(c.strip() for c in r) for r in t.rows))
    out = []
    for n, text in sorted(src.pages.items()):
        out.append(PageInfo(
            doc_id=src.doc_id, part=part, session=session, role=role, page=n,
            words=len(text.split()), numbers=len(numeric_tokens(text)), table_rows=rows_on[n],
            cover=is_cover_page(part, n, text), signature=is_signature_page(text), toc=is_toc(text),
            scan_of=parse_scan_of(role)))
    return out


def eligible(p: PageInfo, qtype: str) -> bool:
    if not p.usable:
        return False
    if qtype == "table":
        return p.table_rows >= MIN_TABLE_ROWS
    if qtype == "calculation":
        return p.table_rows >= MIN_TABLE_ROWS or p.numbers >= MIN_CALC_NUMBERS
    if qtype == "explain":
        return p.words >= MIN_EXPLAIN_WORDS and p.table_rows == 0
    return True


def select_jobs(infos: list[PageInfo], plan: dict[tuple[str, str], int] = PLAN, seed: int = SEED,
                used: dict[tuple[str, int], set[str]] | None = None) -> tuple[list[Job], dict[str, int]]:
    """
    Seeded page choice for the single-page jobs. Returns (jobs, shortfall per "type/bucket").
    A page is used at most once per type; pages no job has used yet are preferred, so pages are
    shared between types only where a bucket runs short (consumer forms: 29 usable pages).
    `used` (page -> types, updated in place) holds pages already taken, e.g. by pairs.py; a scan
    shares its entry with its source page.
    """
    rng = random.Random(seed)
    used = {} if used is None else used
    key = lambda p: p.scan_of or (p.doc_id, p.page)  # noqa: E731
    jobs: list[Job] = []
    short: dict[str, int] = {}
    counter: dict[str, int] = defaultdict(int)
    for (qtype, bucket), want in plan.items():          # dict order = fixed order
        by_session: dict[str, list[PageInfo]] = defaultdict(list)
        for p in infos:
            if p.bucket == bucket and eligible(p, qtype):
                by_session[p.session].append(p)
        sessions = sorted(by_session)
        rng.shuffle(sessions)
        got, i = 0, 0
        while got < want and sessions:
            s = sessions[i % len(sessions)]
            free = [p for p in by_session[s] if qtype not in used.get(key(p), set())]
            if not free:
                sessions.remove(s)                       # exhausted for this type
                continue
            fresh = [p for p in free if not used.get(key(p))]
            p = rng.choice(fresh or free)
            used.setdefault(key(p), set()).add(qtype)
            counter[qtype] += 1
            jobs.append(Job(f"{qtype}-{counter[qtype]:03d}", qtype, bucket, p.session, ((p.doc_id, p.page),)))
            got, i = got + 1, i + 1
        if got < want:
            short[f"{qtype}/{bucket}"] = want - got
            log.warning("%s/%s: %d of %d jobs (not enough eligible pages)", qtype, bucket, got, want)
    return jobs, short


# ---- corpus I/O -----------------------------------------------------------------------------
def load_corpus_infos(refresh: bool = False) -> list[PageInfo]:
    """Page features of the whole corpus (cached in data/results/r3/page_features.json)."""
    if PAGES_CACHE.exists() and not refresh:
        rows = json.loads(PAGES_CACHE.read_text())
        return [PageInfo(**{**r, "scan_of": tuple(r["scan_of"]) if r["scan_of"] else None}) for r in rows]
    from evals.chunking.source import build_source
    from evals.corpus.manifest import read_manifest
    from evals.parsers.config import CONFIGS, downstream_config, load_cached
    from evals.parsers.production_adapter import to_extraction_result
    infos: list[PageInfo] = []
    for row in read_manifest():
        cached = load_cached(CONFIGS[downstream_config("P2", row.part)], row.doc_id)
        if cached is None:
            log.warning("%s: no parse cache", row.doc_id)
            continue
        src = build_source(to_extraction_result(cached, row.doc_id))
        infos += page_infos(src, row.part, row.deal_id, row.role)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_CACHE.write_text(json.dumps([asdict(p) for p in infos]))
    return infos
