"""
Render docs/21_r2_chunk_metrics.md from data/results/r2/chunk_metrics.json (F7: numbers are
generated, never typed by hand).

    venv/bin/python -m evals.chunking.r2_report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.chunking.run_metrics import OUT, REPO

DOC = REPO / "docs" / "21_r2_chunk_metrics.md"

HEAD = """# R2 Results — Chunk Metrics (§8.2)

**Generated** by `venv/bin/python -m evals.chunking.r2_report` from `data/results/r2/chunk_metrics.json`
(run: `venv/bin/python -m evals.chunking.run_metrics`). Do not edit numbers by hand (F7).

Corpus: 92 documents (41 EDGAR, 11 consumer, 40 simulated scans); parse `downstream_config("P2", part)`,
cleaner `{cleaning}`. Definitions: `evals/chunking/metrics.py` docstring and the scope Decision Log
(2026-11-13). Decisions and caveats: `docs/18_session_log.md` (S-2026-11-13-4 … 9).
★ = headline metric, with document-bootstrap 95% CIs (documents as units).
"""

NOTES = """
## Notes and caveats

- **H8** (C1 truncation > 50%, C2–C5 < 5%) is **supported** by these measurements; the formal
  hypothesis write-up is in R8.
- Sentence-break uses C2's splitter (syntok per line) as the reference, so C2 and C5 only break
  sentences they split for size: they are favoured by construction.
- C1 duplicates tables (page text + markdown); its redundancy includes that. Its low table-break
  rate comes from the markdown copy being kept whole.
- C4's never-embedded words are page headers, footers and page numbers, left out on purpose.
- C1 unaligned words (in truncated chunks, not found in the chunk text) count as not embedded;
  counting them as embedded instead changes C1's rate by at most unaligned ÷ page words.
- C4 = Docling headings **plus the run-in heading rule** (S-2026-11-13-10, a recorded deviation
  from §6.1); strict C4 is `C4LayoutAware(runin_headings=False)`.
- C2 overlap is whole sentences only (46% of boundaries have none); C5 is the literal p95 rule
  (89% of its cuts come from the size limit). Both kept as built by decision (S-2026-11-13-11, 12).
- Evidence fragmentation (★) needs the QA set: pending R3.
- Timings are one laptop CPU run (Apple Silicon, 8 cores): rough, not a benchmark.
"""


def _ci(d: dict) -> str:
    if d.get("point") is None:
        return "–"
    if d["lo"] == d["hi"] == d["point"]:
        return f"{d['point']:g}%"
    return f"**{d['point']:g}%** [{d['lo']:g}, {d['hi']:g}]"


def _p(x) -> str:
    return "–" if x is None else f"{x:g}%"


def _dist(xs) -> str:
    return " / ".join(f"{x:,}" for x in xs)


def table(chunkers: dict, rows: list[tuple[str, callable]]) -> str:
    ks = list(chunkers)
    out = ["| | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    out += [f"| {name} | " + " | ".join(f(chunkers[k]) for k in ks) + " |" for name, f in rows]
    return "\n".join(out)


MAIN = [
    ("Chunks", lambda v: f"{v['chunks']:,}"),
    ("MiniLM tokens p5/p50/p95/max", lambda v: _dist(v["size_p5_p50_p95_max"]["minilm"])),
    ("tiktoken p5/p50/p95/max", lambda v: _dist(v["size_p5_p50_p95_max"]["tiktoken"])),
    ("Words p5/p50/p95/max", lambda v: _dist(v["size_p5_p50_p95_max"]["words"])),
    ("★ Truncation rate", lambda v: _ci(v["truncation_rate_pct"])),
    ("Tokens dropped per chunk copy", lambda v: _p(v["tokens_dropped_per_copy_pct"])),
    ("★ Page words never embedded", lambda v: _ci(v["page_words_never_embedded_pct"])),
    ("Unaligned words (C1 alignment gap)", lambda v: f"{v['unaligned_words']:,}"),
    ("Tables (header + data row spans)", lambda v: f"{v['tables']:,}"),
    ("Tables split", lambda v: _p(v["tables_split_pct"])),
    ("Table-break rate (split, header not repeated)", lambda v: _p(v["table_break_rate_pct"])),
    ("Sentence-break rate", lambda v: _p(v["sentence_break_rate_pct"])),
    ("Overlap redundancy", lambda v: _p(v["overlap_redundancy_pct"])),
    ("★ Evidence fragmentation", lambda v: "pending R3"),
]
INDEX = [
    ("Vectors", lambda v: f"{v['index']['vectors']:,}"),
    ("Index memory (MB)", lambda v: f"{v['index']['memory_mb']:g}"),
    ("Index disk (MB)", lambda v: f"{v['index']['disk_mb']:g}"),
    ("Chunking time, total (s)", lambda v: f"{v['chunk_seconds']:g}"),
    ("Embedding time, total (s)", lambda v: f"{v['embed_seconds']:g}"),
    ("Embedding time per doc (s)", lambda v: f"{v['index']['embed_seconds_per_doc']:g}"),
]
PART = [
    ("Chunks", lambda v: f"{v['chunks']:,}"),
    ("★ Truncation rate", lambda v: _ci(v["truncation_rate_pct"])),
    ("★ Page words never embedded", lambda v: _ci(v["page_words_never_embedded_pct"])),
    ("Table-break rate", lambda v: f"{_p(v['table_break_rate_pct'])} (n={v['tables']})"),
    ("Sentence-break rate", lambda v: _p(v["sentence_break_rate_pct"])),
    ("Overlap redundancy", lambda v: _p(v["overlap_redundancy_pct"])),
]


def render(res: dict) -> str:
    ch = res["chunkers"]
    parts = ["## 1. All documents", table(ch, MAIN), "", "## 2. Index size and build time",
             "One FAISS IndexFlatIP per chunker over the corpus; production MiniLM, normalized, batch 64, CPU.",
             "", table(ch, INDEX), "", "## 3. By corpus part"]
    names = sorted({p for v in ch.values() for p in v["by_part"]})
    for p in names:
        sub = {k: v["by_part"][p] for k, v in ch.items() if p in v["by_part"]}
        docs = next(iter(sub.values()))["docs"]
        parts += [f"### {p} ({docs} docs)", table(sub, PART), ""]
    params = ["## 4. Chunker parameters", ""] + [f"- **{k}:** `{json.dumps(v['params'])}`" for k, v in ch.items()]
    return "\n".join([HEAD.format(cleaning=res["cleaning_version"])] + parts + [NOTES] + params) + "\n"


def main() -> int:
    src = OUT / "chunk_metrics.json"
    if not src.exists():
        print(f"missing {src}: run evals.chunking.run_metrics first", file=sys.stderr)
        return 1
    DOC.write_text(render(json.loads(src.read_text())))
    print(f"wrote {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
