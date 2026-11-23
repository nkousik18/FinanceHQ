"""
R3 drafting, step 2 of 3: cross_doc jobs (scope v2 §5.3 "for amendment questions it pairs each
amendment section with the original section it modifies"). No LLM here.

**Amendment pairs (EDGAR).** In each amendment page, find what it changes:
- a section: "Section 5.03 of the Credit Agreement is hereby amended …" (also "Clause (a) of
  Section 5.08 …"); target = the original page whose line starts with that section number
  followed by a title ("SECTION 5.03. Financial Covenant. …"), TOC lines/pages skipped;
- a defined term: 'the definition of "Applicable Rate" … is amended', or a term restated right
  after "Section 1.01 … is amended" ('"Loan Documents" shall mean …'); target = the original
  page with '"Term" means' / '"Term" shall mean'.
Each found (amendment page, original page) is a candidate pair. Pairs whose label names a money
term (margin, rate, commitment, ratio, maturity, fee, …) are preferred, since those make
"value after the amendment" questions with a superseded value; deals in seeded round robin.

**Form pairs (consumer).** A session with a Loan Estimate and a Closing Disclosure gives pairs of
the same form page (LE p2 ↔ CD p2, …): "what changed between estimate and closing" questions.
These are cross_doc but not amendment questions.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from evals.golden.qa.pages import Job, PageInfo, is_toc

SEED = 20261113
AMENDMENT_JOBS = 9
FORM_JOBS = 3
FORM_PAGES = (2, 3)        # first two form pages after the CFPB cover sheet

_SEC = r"(\d{1,2}\.\d{1,2})(?:\([a-z0-9]{1,5}\))*"
_AMENDED = r"(?:is|are|shall\s+be)\s+(?:and\s+hereby\s+is\s+|hereby\s+)?(?:amended|restated|deleted|replaced)"
_GAP = r"(?:[^.;]|(?<=\d)\.(?=\d))"      # within one sentence; a dot inside "1.01" is not an end
SECTION_AMENDED = re.compile(r"Section\s+" + _SEC + _GAP + r"{0,160}?" + _AMENDED, re.S)
DEFINITION_AMENDED = re.compile(r'definitions?\s+of\s+"([^"]{2,80})"' + _GAP + r"{0,200}?" + _AMENDED, re.S)
DEFINES = re.compile(r'^\s*"([^"]{2,80})"\s+(?:shall\s+)?means?\b', re.M)
MONEY_WORDS = re.compile(r"\b(?:margin|rates?|commitments?|ratios?|maturity|fees?|leverage|coverage|"
                         r"pricing|covenants?|amounts?|interest|sofr|libor|termination time)\b", re.I)
SECTION_HEAD = re.compile(r"(?:^|\s)(?:SECTION\s+|Section\s+)?\d{1,2}\.\d{1,2}\.?\s+[A-Z][a-z]")
MAX_HEADS = 15             # TOC pages name ~75 distinct headings; body pages <= 9 on this corpus


def is_contents(text: str) -> bool:
    """TOC page: dot-leader / page-number lines (pages.is_toc) or many section headings."""
    return is_toc(text) or len({h.strip() for h in SECTION_HEAD.findall(text)}) > MAX_HEADS


@dataclass(frozen=True)
class Target:
    kind: str            # "section" | "definition"
    key: str             # "5.03" | "Loan Documents"


@dataclass(frozen=True)
class Pair:
    session: str
    amend: tuple[str, int]
    orig: tuple[str, int]
    label: str           # "Section 5.03 Financial Covenant" | 'definition "Loan Documents"'

    @property
    def money(self) -> bool:
        return bool(MONEY_WORDS.search(self.label))


def amended_targets(text: str) -> list[Target]:
    """What an amendment page changes, in page order, without duplicates."""
    found: list[tuple[int, Target]] = []
    for m in SECTION_AMENDED.finditer(text):
        sec = m.group(1)
        if sec in ("1.01", "1.1"):        # definitions section: the terms are the targets
            tail = text[m.end():m.end() + 3000]
            nxt = SECTION_AMENDED.search(tail)
            for d in DEFINES.finditer(tail[:nxt.start()] if nxt else tail):
                found.append((m.start(), Target("definition", d.group(1))))
        else:
            found.append((m.start(), Target("section", sec)))
    for m in DEFINITION_AMENDED.finditer(text):
        found.append((m.start(), Target("definition", m.group(1))))
    out: list[Target] = []
    for _, t in sorted(found, key=lambda x: x[0]):
        if t not in out:
            out.append(t)
    return out


def find_section(pages: dict[int, str], sec: str) -> tuple[int, str] | None:
    """First non-TOC page with a line starting with section `sec` + a title; (page, title)."""
    head = re.compile(r"^(?:SECTION|Section)?\s*" + re.escape(sec) + r"\.?\s+([A-Z][^\n]{2,120})$", re.M)
    for n in sorted(pages):
        if is_contents(pages[n]):
            continue
        for m in head.finditer(pages[n]):
            line = m.group(1)
            if re.search(r"(?:\.{3,}|\s\d{1,3})\s*$", line):     # TOC-style line
                continue
            title = re.split(r"(?<=[a-z\]])\.\s", line, maxsplit=1)[0].rstrip(".")
            return n, title[:80]
    return None


def find_definition(pages: dict[int, str], term: str) -> int | None:
    pat = re.compile(r'^\s*"' + re.escape(term) + r'"\s+(?:shall\s+)?means?\b', re.M)
    for n in sorted(pages):
        if pat.search(pages[n]) and not is_contents(pages[n]):
            return n
    return None


def amendment_pairs(texts: dict[str, dict[int, str]], roles: dict[str, tuple[str, str]]) -> list[Pair]:
    """
    All amendment ↔ original pairs. `texts`: doc -> page -> cleaned text; `roles`: doc ->
    (session, role). The original is the session's `original` doc.
    """
    original = {s: d for d, (s, r) in roles.items() if r == "original"}
    out: list[Pair] = []
    for doc in sorted(texts):
        session, role = roles[doc]
        if not role.startswith("amendment") or session not in original:
            continue
        orig_doc = original[session]
        orig_pages = texts.get(orig_doc, {})
        seen: set[Target] = set()
        for n in sorted(texts[doc]):
            for t in amended_targets(texts[doc][n]):
                if t in seen:
                    continue
                seen.add(t)
                if t.kind == "section":
                    hit = find_section(orig_pages, t.key)
                    if hit:
                        out.append(Pair(session, (doc, n), (orig_doc, hit[0]), f"Section {t.key} {hit[1]}"))
                else:
                    p = find_definition(orig_pages, t.key)
                    if p:
                        out.append(Pair(session, (doc, n), (orig_doc, p), f'definition "{t.key}"'))
    return out


def form_pairs(infos: list[PageInfo]) -> list[Pair]:
    """LE ↔ CD pairs of the same form page within a consumer session."""
    by = {(p.doc_id, p.page): p for p in infos if p.part == "consumer"}
    docs: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"le": set(), "cd": set()})
    for p in by.values():
        kind = "le" if "_le_" in p.doc_id else "cd" if "_cd_" in p.doc_id else None
        if kind:
            docs[p.session][kind].add(p.doc_id)
    out = []
    for session in sorted(docs):
        for le in sorted(docs[session]["le"]):
            for cd in sorted(docs[session]["cd"]):
                for n in FORM_PAGES:
                    if (le, n) in by and (cd, n) in by and by[(le, n)].usable and by[(cd, n)].usable:
                        out.append(Pair(session, (le, n), (cd, n), f"Loan Estimate vs Closing Disclosure, form page {n - 1}"))
    return out


def _round_robin(pairs: list[Pair], want: int, rng: random.Random) -> list[Pair]:
    by: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by[p.session].append(p)
    for s in sorted(by):
        rng.shuffle(by[s])
    sessions = sorted(by)
    rng.shuffle(sessions)
    out: list[Pair] = []
    while len(out) < want and any(by.values()):
        for s in sessions:
            if by[s] and len(out) < want:
                out.append(by[s].pop(0))
    return out


def _spread(pairs: list[Pair], want: int, rng: random.Random) -> list[Pair]:
    """Money pairs first (seeded round robin over sessions), then the rest the same way.
    An amendment page that changes several targets is used once."""
    money = _round_robin([p for p in pairs if p.money], len(pairs), rng)
    rest = _round_robin([p for p in pairs if not p.money], len(pairs), rng)
    out: list[Pair] = []
    for p in money + rest:
        if len(out) < want and all(p.amend != q.amend and p.orig != q.orig for q in out):
            out.append(p)
    return out


def select_pair_jobs(amend: list[Pair], forms: list[Pair], seed: int = SEED,
                     used: dict[tuple[str, int], set[str]] | None = None,
                     n_amend: int = AMENDMENT_JOBS, n_form: int = FORM_JOBS) -> tuple[list[Job], dict[str, int]]:
    """cross_doc jobs (amendment first). Marks their pages used for "cross_doc" in `used`."""
    rng = random.Random(seed)
    used = {} if used is None else used
    jobs, short = [], {}
    chosen = [(p, True) for p in _spread(amend, n_amend, rng)] + [(p, False) for p in _spread(forms, n_form, rng)]
    for i, (p, is_amend) in enumerate(chosen, 1):
        jobs.append(Job(f"cross_doc-{i:03d}", "cross_doc", "edgar" if is_amend else "consumer", p.session,
                        (p.amend, p.orig), note=p.label, amendment=is_amend))
        for page in (p.amend, p.orig):
            used.setdefault(page, set()).add("cross_doc")
    if (n := sum(a for _, a in chosen)) < n_amend:
        short["cross_doc/amendment"] = n_amend - n
    if (n := sum(not a for _, a in chosen)) < n_form:
        short["cross_doc/form"] = n_form - n
    return jobs, short
