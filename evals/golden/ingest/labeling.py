"""
Ingestion golden set: page selection, Docling pre-fill, images, validation (scope v2 §5.4).

    venv/bin/python -m evals.golden.ingest.labeling select     # choose 30 pages (seeded; Docling-independent)
    venv/bin/python -m evals.golden.ingest.labeling prefill    # label files from Docling P1 (skips existing files)
    venv/bin/python -m evals.golden.ingest.labeling render [page_id ...]   # images + overlays
    venv/bin/python -m evals.golden.ingest.labeling validate   # F4 checks on every label file
    venv/bin/python -m evals.golden.ingest.labeling status

Committed:  selection.json, pages/<page_id>.json, images/<page_id>.png
Gitignored: data/golden_overlays/<page_id>.png (page + 50pt grid + current label boxes)

How to label: see README.md in this folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import fitz  # PyMuPDF

from evals.corpus.cfpb import is_cover_page
from evals.corpus.manifest import read_manifest
from evals.corpus.scans import is_signature_page
from evals.golden.ingest.schema import validate_labels
from evals.parsers.docling_doc import page_view
from evals.parsers.config import P1, load_cached

log = logging.getLogger("evals.golden.ingest")

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
IMAGES = HERE / "images"
SELECTION = HERE / "selection.json"
OVERLAYS = REPO / "data" / "golden_overlays"
PDF_DIR = REPO / "data" / "corpus" / "pdf"
SCAN_SPECS = REPO / "evals" / "corpus" / "scan_specs.json"

SEED = 20261105
MIN_WORDS = 80
QUOTAS = {"digital_prose": 10, "table_heavy_edgar": 5, "table_heavy_consumer": 3,
          "scanned": 6, "complex_layout": 6}
FROM_SCRATCH = {"digital_prose": 2, "table_heavy": 1, "scanned": 1, "complex_layout": 1}
IMAGE_DPI = 100
MAX_PER_CONSUMER_DOC = 2   # 29 non-cover CFPB pages in 11 forms

CLASS_COLORS = {
    "title": (0.8, 0.0, 0.0), "section_header": (0.85, 0.35, 0.0), "text": (0.0, 0.35, 0.8),
    "list": (0.0, 0.55, 0.55), "table": (0.0, 0.6, 0.0), "page_header": (0.5, 0.0, 0.6),
    "page_footer": (0.5, 0.0, 0.6), "page_number": (0.7, 0.0, 0.7), "other": (0.4, 0.4, 0.4),
}


def page_id(doc_id: str, page: int) -> str:
    return f"{doc_id}_p{page:03d}"


# ------------------------------------------------------------------
# Selection (independent of Docling output, so the set isn't biased toward it)
# ------------------------------------------------------------------

def _has_table(page: fitz.Page) -> bool:
    try:
        return any(t.row_count >= 3 for t in page.find_tables().tables)
    except Exception:
        return False


def select(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows = {r.doc_id: r for r in read_manifest()}
    specs = json.loads(SCAN_SPECS.read_text())["pages"]

    # 1. Scanned: 2 per degradation level.
    chosen: list[dict] = []
    used_sources: set[tuple[str, int]] = set()
    for level in ("light", "medium", "heavy"):
        pool = [s for s in specs if s["level"] == level]
        rng.shuffle(pool)
        for s in pool[:2]:
            chosen.append({"doc_id": s["scan_id"], "page": 1, "stratum": "scanned",
                           "scan_of": f"{s['source_doc']}:p{s['page']}", "scan_level": level})
            used_sources.add((s["source_doc"], s["page"]))

    # 2. Digital pages: walk (doc, page) in seeded order, fill quotas.
    cands = [(r, i) for r in rows.values() if r.part in ("edgar", "consumer")
             for i in range(int(r.pdf_pages or 0))]
    rng.shuffle(cands)
    need = {k: v for k, v in QUOTAS.items() if k != "scanned"}
    docs_used: dict[str, set[str]] = {}          # stratum -> docs (1 page per doc per stratum)
    consumer_docs: dict[str, int] = {}           # CFPB: pages taken per form, all strata
    open_docs: dict[str, fitz.Document] = {}
    try:
        for row, idx in cands:
            if not any(need.values()):
                break
            if (row.doc_id, idx + 1) in used_sources:
                continue
            if row.part == "consumer" and consumer_docs.get(row.doc_id, 0) >= MAX_PER_CONSUMER_DOC:
                continue
            doc = open_docs.setdefault(row.doc_id, fitz.open(PDF_DIR / f"{row.doc_id}.pdf"))
            page = doc[idx]
            text = page.get_text()
            if len(text.split()) < MIN_WORDS or is_signature_page(text) \
                    or is_cover_page(row.part, idx + 1, text):
                continue
            table = _has_table(page)
            if row.part == "edgar":
                bucket = "table_heavy_edgar" if table else "digital_prose"
            else:
                # CFPB form pages are complex layouts with or without ruled tables:
                # table pages fill the consumer table quota first, the rest go to complex_layout.
                bucket = "table_heavy_consumer" if table and need["table_heavy_consumer"] > 0 \
                    else "complex_layout"
            if need.get(bucket, 0) <= 0 or row.doc_id in docs_used.get(bucket, set()):
                continue
            need[bucket] -= 1
            docs_used.setdefault(bucket, set()).add(row.doc_id)
            if row.part == "consumer":
                consumer_docs[row.doc_id] = consumer_docs.get(row.doc_id, 0) + 1
            stratum = "table_heavy" if bucket.startswith("table_heavy") else bucket
            chosen.append({"doc_id": row.doc_id, "page": idx + 1, "stratum": stratum})
    finally:
        for d in open_docs.values():
            d.close()
    short = {k: v for k, v in need.items() if v > 0}

    # 3. From-scratch pages: the first N of each stratum in a seeded order.
    by_stratum: dict[str, list[dict]] = {}
    for c in chosen:
        by_stratum.setdefault(c["stratum"], []).append(c)
    for stratum, items in by_stratum.items():
        items.sort(key=lambda c: (c["doc_id"], c["page"]))
        order = items[:]
        rng.shuffle(order)
        for c in order[: FROM_SCRATCH.get(stratum, 0)]:
            c["from_scratch"] = True
    for c in chosen:
        c.setdefault("from_scratch", False)
        c["page_id"] = page_id(c["doc_id"], c["page"])
        c["source_format"] = rows[c["doc_id"]].source_format

    chosen.sort(key=lambda c: (c["stratum"], c["page_id"]))
    SELECTION.write_text(json.dumps({"seed": seed, "quotas": QUOTAS, "from_scratch": FROM_SCRATCH,
                                     "min_words": MIN_WORDS, "quota_shortfall": short,
                                     "pages": chosen}, indent=2))
    if short:
        log.warning("quota shortfall: %s", short)
    return chosen


def replace(drop: list[str], reason: str, seed: int = SEED) -> dict[str, str]:
    """
    Swap selected pages for new ones of the same stratum, leaving every other page (and its
    from-scratch flag) untouched. Replays select()'s seeded candidate order and takes the next
    eligible page: same filters as select(), one page per doc per stratum, and never an
    image filing (its text layer is invisible white text; S-2026-11-12-13).
    Records each swap under "replaced" in selection.json. Returns {old_page_id: new_page_id}.
    """
    sel = json.loads(SELECTION.read_text())
    rows = {r.doc_id: r for r in read_manifest()}
    image_filings = {d for d, r in rows.items() if "images_embedded" in (r.renderer or "")}
    pages = sel["pages"]
    by_id = {s["page_id"]: s for s in pages}
    for pid in drop:
        if pid not in by_id:
            raise KeyError(pid)
        if by_id[pid]["stratum"] != "digital_prose":
            raise ValueError(f"{pid}: only digital_prose replacement is implemented")

    # Same RNG sequence as select(): scanned pools first, then the candidate shuffle.
    rng = random.Random(seed)
    for level in ("light", "medium", "heavy"):
        pool = [s for s in json.loads(SCAN_SPECS.read_text())["pages"] if s["level"] == level]
        rng.shuffle(pool)
    cands = [(r, i) for r in rows.values() if r.part in ("edgar", "consumer")
             for i in range(int(r.pdf_pages or 0))]
    rng.shuffle(cands)

    keep = [s for s in pages if s["page_id"] not in drop]
    taken = {(s["doc_id"], s["page"]) for s in keep}
    scan_sources = {tuple(s["scan_of"].split(":p")) for s in keep if s.get("scan_of")}
    used_docs = {s["doc_id"] for s in keep if s["stratum"] == "digital_prose"}
    swaps: dict[str, str] = {}
    todo = list(drop)
    open_docs: dict[str, fitz.Document] = {}
    try:
        for row, idx in cands:
            if not todo:
                break
            if row.part != "edgar" or row.doc_id in image_filings or row.doc_id in used_docs:
                continue
            if (row.doc_id, idx + 1) in taken or (row.doc_id, str(idx + 1)) in scan_sources:
                continue
            doc = open_docs.setdefault(row.doc_id, fitz.open(PDF_DIR / f"{row.doc_id}.pdf"))
            page = doc[idx]
            text = page.get_text()
            if len(text.split()) < MIN_WORDS or is_signature_page(text) \
                    or is_cover_page(row.part, idx + 1, text) or _has_table(page):
                continue
            old = todo.pop(0)
            new = {"doc_id": row.doc_id, "page": idx + 1, "stratum": "digital_prose",
                   "from_scratch": by_id[old]["from_scratch"], "page_id": page_id(row.doc_id, idx + 1),
                   "source_format": row.source_format}
            pages[pages.index(by_id[old])] = new
            used_docs.add(row.doc_id)
            swaps[old] = new["page_id"]
    finally:
        for d in open_docs.values():
            d.close()
    if todo:
        raise RuntimeError(f"no replacement found for {todo}")
    pages.sort(key=lambda c: (c["stratum"], c["page_id"]))
    sel.setdefault("replaced", []).extend({"old": o, "new": n, "reason": reason} for o, n in swaps.items())
    SELECTION.write_text(json.dumps(sel, indent=2))
    return swaps


# ------------------------------------------------------------------
# Docling -> label pre-fill (shared reader: evals/parsers/docling_doc.py)
# ------------------------------------------------------------------

def prefill_from_docling(doc: dict, page: int) -> dict:
    view = page_view(doc, page)
    elements = [{"id": e["id"], "class": e["class"], "bbox": e["bbox"], "text": e["text"][:300]}
                for e in view.elements]
    return {"layout_elements": elements, "reading_order": view.reading_order,
            "tables": view.tables, "header_footer_lines": view.header_footer_lines}


def blank_labels(sel: dict, page_size: list[float]) -> dict:
    scanned = sel["stratum"] == "scanned"
    return {
        "page_id": sel["page_id"], "doc_id": sel["doc_id"], "page": sel["page"],
        "stratum": sel["stratum"], "source_format": sel["source_format"],
        "scan_of": sel.get("scan_of"), "scan_level": sel.get("scan_level"),
        "page_size": page_size, "status": "prefilled", "from_scratch": sel["from_scratch"],
        "prefill_source": None, "labeled_by": None, "verified_on": None,
        "reference_text_source": "source_page_text_layer" if scanned else "text_layer",
        "reference_text": None,
        "layout_elements": [], "reading_order": [], "tables": [], "key_values": [],
        "header_footer_lines": [], "notes": "",
    }


def prefill(overwrite: bool = False) -> dict:
    sel = json.loads(SELECTION.read_text())["pages"]
    PAGES.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "kept": 0, "no_cache": []}
    for s in sel:
        out = PAGES / f"{s['page_id']}.json"
        if out.exists() and not overwrite:
            stats["kept"] += 1           # never clobber Kousik's edits
            continue
        with fitz.open(PDF_DIR / f"{s['doc_id']}.pdf") as d:
            rect = d[s["page"] - 1].rect
        lab = blank_labels(s, [round(rect.width, 1), round(rect.height, 1)])
        if not s["from_scratch"]:
            cached = load_cached(P1, s["doc_id"])
            if cached is None:
                stats["no_cache"].append(s["doc_id"])
                continue
            lab.update(prefill_from_docling(cached["document"], s["page"]))
            lab["prefill_source"] = f"docling {cached['meta']['versions']['docling']} P1"
        out.write_text(json.dumps(lab, indent=2, ensure_ascii=False))
        stats["written"] += 1
    return stats


# ------------------------------------------------------------------
# Images
# ------------------------------------------------------------------

def render(page_ids: list[str] | None = None) -> int:
    sel = {s["page_id"]: s for s in json.loads(SELECTION.read_text())["pages"]}
    IMAGES.mkdir(parents=True, exist_ok=True)
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    n = 0
    for pid in page_ids or sorted(sel):
        s = sel[pid]
        with fitz.open(PDF_DIR / f"{s['doc_id']}.pdf") as d:
            page = d[s["page"] - 1]
            page.get_pixmap(dpi=IMAGE_DPI).save(IMAGES / f"{pid}.png")

            # Overlay: draw on a copy so the source PDF is untouched.
            tmp = fitz.open()
            tmp.insert_pdf(d, from_page=s["page"] - 1, to_page=s["page"] - 1)
            p = tmp[0]
            # Chrome-rendered pages leave a scaling transform active at the end of
            # their content stream; wrap it so our drawing uses plain page coordinates.
            p.wrap_contents()
            w, h = p.rect.width, p.rect.height
            for x in range(0, int(w) + 1, 50):
                p.draw_line((x, 0), (x, h), color=(0.75, 0.75, 0.75), width=0.3)
                p.insert_text((x + 1, 7), str(x), fontsize=5, color=(0.5, 0.5, 0.5))
            for y in range(0, int(h) + 1, 50):
                p.draw_line((0, y), (w, y), color=(0.75, 0.75, 0.75), width=0.3)
                p.insert_text((1, y - 1), str(y), fontsize=5, color=(0.5, 0.5, 0.5))
            lab_path = PAGES / f"{pid}.json"
            if lab_path.exists():
                lab = json.loads(lab_path.read_text())
                for e in lab["layout_elements"]:
                    color = CLASS_COLORS.get(e.get("class"), (0, 0, 0))
                    r = fitz.Rect(*e["bbox"])
                    p.draw_rect(r, color=color, width=0.8)
                    p.insert_text((r.x0 + 1, r.y0 + 6), f"{e['id']} {e.get('class')}", fontsize=5.5, color=color)
            p.get_pixmap(dpi=150).save(OVERLAYS / f"{pid}.png")
            tmp.close()
        n += 1
    return n


# ------------------------------------------------------------------
# Status / validate
# ------------------------------------------------------------------

def validate_all() -> dict[str, list[str]]:
    out = {}
    for path in sorted(PAGES.glob("*.json")):
        errs = validate_labels(json.loads(path.read_text()))
        if errs:
            out[path.stem] = errs
    return out


def status() -> dict:
    sel = json.loads(SELECTION.read_text())["pages"]
    rows = []
    for s in sel:
        path = PAGES / f"{s['page_id']}.json"
        lab = json.loads(path.read_text()) if path.exists() else None
        rows.append({"page_id": s["page_id"], "stratum": s["stratum"], "from_scratch": s["from_scratch"],
                     "status": lab["status"] if lab else "missing",
                     "elements": len(lab["layout_elements"]) if lab else 0,
                     "tables": len(lab["tables"]) if lab else 0,
                     "problems": len(validate_labels(lab)) if lab else None})
    return {"pages": rows,
            "verified": sum(r["status"] == "verified" for r in rows), "total": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    rp = sub.add_parser("replace", help="swap selected pages for new ones of the same stratum")
    rp.add_argument("page_ids", nargs="+")
    rp.add_argument("--reason", required=True)
    pf = sub.add_parser("prefill")
    pf.add_argument("--overwrite", action="store_true", help="replace existing label files (loses edits!)")
    rd = sub.add_parser("render")
    rd.add_argument("page_ids", nargs="*")
    sub.add_parser("validate")
    sub.add_parser("status")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "select":
        from collections import Counter
        chosen = select()
        print(json.dumps({"pages": len(chosen), "by_stratum": Counter(c["stratum"] for c in chosen),
                          "from_scratch": sum(c["from_scratch"] for c in chosen)}, indent=2))
    elif args.cmd == "replace":
        swaps = replace(args.page_ids, args.reason)
        for old in swaps:
            for path in (PAGES / f"{old}.json", IMAGES / f"{old}.png", OVERLAYS / f"{old}.png"):
                path.unlink(missing_ok=True)      # old labels stay in git history
        print(json.dumps(swaps, indent=2))
        print("next: labeling prefill && labeling render <new page_ids>")
    elif args.cmd == "prefill":
        print(json.dumps(prefill(args.overwrite), indent=2))
    elif args.cmd == "render":
        print(f"rendered {render(args.page_ids or None)} pages; overlays in {OVERLAYS}")
    elif args.cmd == "validate":
        problems = validate_all()
        for pid, errs in problems.items():
            print(pid)
            for e in errs:
                print("   -", e)
        print(f"{len(problems)} file(s) with problems")
        return 1 if problems else 0
    elif args.cmd == "status":
        st = status()
        for r in st["pages"]:
            flag = " (from scratch)" if r["from_scratch"] else ""
            print(f"{r['page_id']:24} {r['stratum']:15} {r['status']:9} el={r['elements']:3} "
                  f"tables={r['tables']} problems={r['problems']}{flag}")
        print(f"verified {st['verified']}/{st['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
