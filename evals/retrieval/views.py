"""
Which documents (and pages) a question's session index holds (R4; Decision Log 2026-11-13).

Each simulated scan is a one-page image copy of a page in its session (manifest role
`scan_of:<doc_id>:p<page>`). Indexing a scan next to its source page would duplicate the text,
and swapping every scan in for good would hide pages that 15 digital questions cite. So:

- Digital view (default): every non-scan document in `session_docs`.
- Scan view (the question cites a scan): the digital view with each cited scan's source page
  dropped and the scan document added. A source page the question also cites directly is not
  dropped (logged; none in the v1 set).

Views are (doc_id, drop_pages) pairs; the index cache keys on both.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)
_SCAN_OF = re.compile(r"^scan_of:(?P<doc>[^:]+):p(?P<page>\d+)$")


@dataclass(frozen=True)
class DocView:
    doc_id: str
    drop_pages: frozenset[int] = frozenset()


def scan_source(role: str) -> tuple[str, int] | None:
    m = _SCAN_OF.match(role or "")
    return (m["doc"], int(m["page"])) if m else None


def session_view(q: dict, part_of: dict[str, str], role_of: dict[str, str]) -> list[DocView]:
    """The documents a question's index is built from, in session_docs order."""
    cited = {e["doc_id"] for e in q["evidence"]}
    cited_pages = {(e["doc_id"], e["page"]) for e in q["evidence"]}
    scans = [d for d in q["session_docs"] if part_of.get(d) == "simulated_scan" and d in cited]
    drop: dict[str, set[int]] = {}
    for s in scans:
        src = scan_source(role_of.get(s, ""))
        if src is None:
            raise ValueError(f"{q['qid']}: scan {s} has no scan_of role ({role_of.get(s)!r})")
        if src in cited_pages:
            log.warning("%s cites both %s and its source %s p%d; source page kept", q["qid"], s, *src)
            continue
        drop.setdefault(src[0], set()).add(src[1])
    out = []
    for d in q["session_docs"]:
        if part_of.get(d) == "simulated_scan":
            if d in scans:
                out.append(DocView(d))
        else:
            out.append(DocView(d, frozenset(drop.get(d, ()))))
    return out
