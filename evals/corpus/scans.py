"""
Simulated scans (scope v2 §5.0): 40 pages from the corpus, rasterized and
degraded into image-only PDFs.

Why simulated: real scans have no exact reference text. Each simulated page
comes from a digital page whose text layer is the reference, so CER and
numeric fidelity can be measured exactly (H6). They are labelled
`simulated_scan` everywhere, and no claim about real-scan accuracy rests on them.

Selection (deterministic, seed recorded):
  edgar_prose 12 · edgar_table 8 · edgar_amendment 8 · consumer 12
  - no CFPB cover sheets (page 1 of every sample PDF)
  - only pages with >= MIN_WORDS words of text, and no signature pages
    (>= 4 "By:" / "Name:" / "Title:" markers): they are mostly blank and say
    little about reading accuracy
  - at most MAX_PER_DOC_STRATUM pages per source document per stratum, and
    MAX_PER_DOC per document overall (tables cluster in a few agreements:
    33 detectable table pages sit in 8 of 41 EDGAR documents)
  - "table" = PyMuPDF finds a table with >= 3 rows on the page. Only ruled
    tables are detected; unruled ones count as prose here
Degradation levels are assigned round-robin within each stratum, so every
stratum gets a mix of light / medium / heavy.

    venv/bin/python -m evals.corpus.scans build

Writes data/corpus/pdf/scan_NN.pdf (gitignored), evals/corpus/scan_specs.json
(committed: every page's source, level and parameters) and manifest rows.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter

from evals.corpus.cfpb import is_cover_page
from evals.corpus.manifest import DocRow, read_manifest, write_manifest

log = logging.getLogger("evals.corpus.scans")

REPO = Path(__file__).resolve().parents[2]
PDF_DIR = REPO / "data" / "corpus" / "pdf"
SPECS = REPO / "evals" / "corpus" / "scan_specs.json"

SEED = 20261104
MIN_WORDS = 80
MAX_PER_DOC = 3
MAX_PER_DOC_STRATUM = 2
QUOTAS = {"edgar_prose": 12, "edgar_table": 8, "edgar_amendment": 8, "consumer": 12}


@dataclass(frozen=True)
class Degradation:
    dpi: int
    blur_radius: float
    noise_sigma: float      # gaussian noise, in 0–255 grey levels
    max_skew_deg: float     # actual skew drawn uniformly from ±max
    jpeg_quality: int
    paper_grey: int         # background level (255 = white)


LEVELS: dict[str, Degradation] = {
    "light":  Degradation(dpi=300, blur_radius=0.3, noise_sigma=4.0,  max_skew_deg=0.3, jpeg_quality=85, paper_grey=250),
    "medium": Degradation(dpi=200, blur_radius=0.7, noise_sigma=10.0, max_skew_deg=1.0, jpeg_quality=60, paper_grey=242),
    "heavy":  Degradation(dpi=150, blur_radius=1.1, noise_sigma=18.0, max_skew_deg=2.0, jpeg_quality=35, paper_grey=232),
}
LEVEL_ORDER = ("light", "medium", "heavy")


# ------------------------------------------------------------------
# Degradation
# ------------------------------------------------------------------

def rasterize(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def degrade(img: Image.Image, d: Degradation, rng: random.Random) -> tuple[bytes, dict]:
    """Apply skew, paper tone, blur, noise and JPEG compression. Returns (jpeg bytes, applied params)."""
    skew = rng.uniform(-d.max_skew_deg, d.max_skew_deg)
    out = img.rotate(skew, resample=Image.BICUBIC, expand=False, fillcolor=255)

    arr = np.asarray(out, dtype=np.float32)
    arr = arr * (d.paper_grey / 255.0)                       # darken paper slightly
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(d.blur_radius))

    noise_rng = np.random.default_rng(rng.randrange(2**32))
    arr = np.asarray(out, dtype=np.float32) + noise_rng.normal(0, d.noise_sigma, size=(out.height, out.width))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=d.jpeg_quality, optimize=False)
    return buf.getvalue(), {"skew_deg": round(skew, 3)}


def image_pdf(jpeg: bytes, page_rect: fitz.Rect, out: Path) -> None:
    """One-page PDF containing only the image (no text layer)."""
    doc = fitz.open()
    page = doc.new_page(width=page_rect.width, height=page_rect.height)
    page.insert_image(page.rect, stream=jpeg)
    doc.set_metadata({})                                     # no timestamps
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out, garbage=4, deflate=True, no_new_id=True)
    doc.close()


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------

_SIG_RE = re.compile(r"\b(?:By|Name|Title):\s")


def is_signature_page(text: str) -> bool:
    return len(_SIG_RE.findall(text)) >= 4


def _has_table(page: fitz.Page) -> bool:
    try:
        return any(t.row_count >= 3 for t in page.find_tables().tables)
    except Exception:
        return False


def stratum_of(row: DocRow, page: fitz.Page) -> str:
    if row.part == "consumer":
        return "consumer"
    if row.role.startswith("amendment"):
        return "edgar_amendment"
    return "edgar_table" if _has_table(page) else "edgar_prose"


def select_pages(rows: list[DocRow], seed: int = SEED, quotas: dict[str, int] = QUOTAS,
                 pdf_dir: Path = PDF_DIR) -> list[dict]:
    """
    Walk (doc, page) pairs in seeded random order and fill each stratum's quota.
    Table detection only runs on candidate pages, so the walk stays cheap.
    """
    rng = random.Random(seed)
    candidates = [(r, i) for r in rows if r.part in ("edgar", "consumer")
                  for i in range(int(r.pdf_pages or 0))]
    rng.shuffle(candidates)

    need = dict(quotas)
    per_doc: dict[str, int] = {}
    per_doc_stratum: dict[tuple[str, str], int] = {}
    picked: list[dict] = []
    open_docs: dict[str, fitz.Document] = {}
    try:
        for row, idx in candidates:
            if not any(need.values()):
                break
            if per_doc.get(row.doc_id, 0) >= MAX_PER_DOC:
                continue
            doc = open_docs.setdefault(row.doc_id, fitz.open(pdf_dir / f"{row.doc_id}.pdf"))
            page = doc[idx]
            text = page.get_text()
            if len(text.split()) < MIN_WORDS or is_signature_page(text) \
                    or is_cover_page(row.part, idx + 1, text):
                continue
            stratum = stratum_of(row, page)
            if need.get(stratum, 0) <= 0:
                continue
            if per_doc_stratum.get((row.doc_id, stratum), 0) >= MAX_PER_DOC_STRATUM:
                continue
            need[stratum] -= 1
            per_doc[row.doc_id] = per_doc.get(row.doc_id, 0) + 1
            per_doc_stratum[(row.doc_id, stratum)] = per_doc_stratum.get((row.doc_id, stratum), 0) + 1
            picked.append({"source_doc": row.doc_id, "page": idx + 1, "stratum": stratum,
                           "session": row.deal_id, "source_url": row.source_url})
    finally:
        for d in open_docs.values():
            d.close()

    short = {k: v for k, v in need.items() if v > 0}
    if short:
        log.warning("quota not filled: %s", short)

    # Round-robin degradation levels within each stratum.
    counters: dict[str, int] = {}
    for p in sorted(picked, key=lambda p: (p["stratum"], p["source_doc"], p["page"])):
        n = counters.get(p["stratum"], 0)
        p["level"] = LEVEL_ORDER[n % len(LEVEL_ORDER)]
        counters[p["stratum"]] = n + 1
    return sorted(picked, key=lambda p: (p["stratum"], p["source_doc"], p["page"]))


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------

def build(seed: int = SEED) -> list[dict]:
    rows = read_manifest()
    picked = select_pages(rows, seed)
    rng = random.Random(seed + 1)

    specs, scan_rows = [], []
    for n, p in enumerate(picked, start=1):
        scan_id = f"scan_{n:02d}"
        d = LEVELS[p["level"]]
        with fitz.open(PDF_DIR / f"{p['source_doc']}.pdf") as src:
            page = src[p["page"] - 1]
            jpeg, applied = degrade(rasterize(page, d.dpi), d, rng)
            image_pdf(jpeg, page.rect, PDF_DIR / f"{scan_id}.pdf")
        spec = {"scan_id": scan_id, **p, "params": {**asdict(d), **applied},
                "image_sha256": hashlib.sha256(jpeg).hexdigest()}
        specs.append(spec)
        scan_rows.append(DocRow(
            doc_id=scan_id, part="simulated_scan", deal_id=p["session"],
            role=f"scan_of:{p['source_doc']}:p{p['page']}", source_url=p["source_url"],
            source_format="image_pdf", source_sha256=spec["image_sha256"], pdf_pages="1",
            renderer=f"pymupdf {fitz.VersionBind} + pillow; level={p['level']}",
        ))

    others = [r for r in rows if r.part != "simulated_scan"]
    write_manifest(others + scan_rows)
    SPECS.write_text(json.dumps({"seed": seed, "quotas": QUOTAS, "min_words": MIN_WORDS,
                                 "max_per_doc": MAX_PER_DOC,
                                 "max_per_doc_stratum": MAX_PER_DOC_STRATUM,
                                 "levels": {k: asdict(v) for k, v in LEVELS.items()},
                                 "pages": specs}, indent=2))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    specs = build(args.seed)
    from collections import Counter
    print(json.dumps({"pages": len(specs),
                      "by_stratum": Counter(s["stratum"] for s in specs),
                      "by_level": Counter(s["level"] for s in specs),
                      "source_docs": len({s["source_doc"] for s in specs})}, indent=2))


if __name__ == "__main__":
    main()
