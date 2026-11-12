"""
HTML -> PDF rendering with headless Chrome, at fixed settings (scope v2 D10).

Settings (recorded in the Decision Log):
- US Letter, 0.5in margins, no browser header/footer
- page fidelity: the source HTML marks its original page breaks. Each
  document is rendered at zoom 1.0, 0.9, 0.8, 0.7 until the PDF has no more
  pages than the source (breaks + 1), so original pages don't spill over.
  The zoom used is recorded per document
- the HTML is sanitized first: <script> removed, <img> removed (credit
  agreements rarely have images; loading them would hit EDGAR without our
  User-Agent), and a Content-Security-Policy blocks every network load
- exception (2026-11-12, S-2026-11-12-13): **image filings** — documents with at least
  IMAGE_FILING_MIN_IMAGES <img> tags — are page scans on EDGAR with an invisible white
  text layer. Stripping their images left blank pages. Their images are fetched by our
  SEC client (User-Agent, throttled, cached) and embedded as data: URIs; the CSP allows
  data: images only, and the images are scaled to fit the printable area
- Chrome path from CHROME_PATH, defaulting to the macOS install
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
import subprocess
import tempfile
from pathlib import Path

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MARGIN_IN = 0.5
ZOOM = 1.0
ZOOM_STEPS = (1.0, 0.9, 0.8, 0.7)
IMAGE_FILING_MIN_IMAGES = 10
_IMG_SRC_RE = re.compile(r"""(<img\b[^>]*?\bsrc\s*=\s*)(["']?)([^"'\s>]+)\2""", re.IGNORECASE)
_FIT_IMAGES_CSS = ("img { max-width: 100% !important; max-height: 9.6in !important; width: auto !important; "
                   "height: auto !important; } body div { width: auto !important; max-width: 100%; }")
_BREAK_RE = re.compile(r"(?:page-)?break-(?:before|after)\s*:\s*(?:always|page)", re.IGNORECASE)


def print_head(margin_in: float = MARGIN_IN, zoom: float = ZOOM, images: bool = False) -> str:
    csp = "default-src 'none'; style-src 'unsafe-inline'" + ("; img-src data:" if images else "")
    fit = _FIT_IMAGES_CSS if images else ""
    return (
        '<meta charset="utf-8">'
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{csp}\">"
        f"<style>@page {{ size: Letter; margin: {margin_in}in; }} body {{ margin: 0; zoom: {zoom}; }}{fit}</style>"
    )


def image_sources(raw: bytes) -> list[str]:
    """Distinct <img src> values in document order."""
    seen: dict[str, None] = {}
    for m in _IMG_SRC_RE.finditer(raw.decode("utf-8", errors="replace")):
        seen.setdefault(m.group(3), None)
    return list(seen)


def is_image_filing(raw: bytes) -> bool:
    return len(image_sources(raw)) >= IMAGE_FILING_MIN_IMAGES


def _data_uri(src: str, data: bytes) -> str:
    mime = mimetypes.guess_type(src)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class RenderError(RuntimeError):
    pass


def chrome_path() -> str:
    path = os.environ.get("CHROME_PATH", DEFAULT_CHROME)
    if not Path(path).exists():
        raise RenderError(f"Chrome not found at {path}; set CHROME_PATH")
    return path


def chrome_version() -> str:
    out = subprocess.run([chrome_path(), "--version"], capture_output=True, text=True, check=True)
    return "chrome " + out.stdout.strip().split()[-1]


def sanitize_html(raw: bytes, margin_in: float = MARGIN_IN, zoom: float = ZOOM,
                  images: dict[str, bytes] | None = None) -> tuple[str, dict]:
    """
    Strip scripts, add print settings. Returns (html, stats).
    `images` None: strip every <img> (default). `images` {src: bytes}: embed those images as
    data: URIs (image filings); any <img> whose src isn't in the dict is still stripped.
    """
    html = raw.decode("utf-8", errors="replace")
    n_img = len(re.findall(r"<img\b", html, re.I))
    stats = {"scripts_removed": len(re.findall(r"<script\b", html, re.I))}
    html = re.sub(r"<script\b.*?</script\s*>", "", html, flags=re.I | re.S)
    if images:
        html = _IMG_SRC_RE.sub(lambda m: (m.group(1) + '"' + _data_uri(m.group(3), images[m.group(3)]) + '"')
                               if m.group(3) in images else m.group(0), html)
        html = re.sub(r"<img\b(?![^>]*\bsrc\s*=\s*[\"']?data:)[^>]*>", "", html, flags=re.I)
        stats["images_embedded"] = len(re.findall(r"<img\b", html, re.I))
        stats["images_removed"] = n_img - stats["images_embedded"]
    else:
        html = re.sub(r"<img\b[^>]*>", "", html, flags=re.I)
        stats["images_removed"] = n_img
    head = print_head(margin_in, zoom, images=bool(images))
    if re.search(r"<head\b[^>]*>", html, re.I):
        html = re.sub(r"(<head\b[^>]*>)", lambda m: m.group(1) + head, html, count=1, flags=re.I)
    else:
        html = head + html
    return html, stats


def render_pdf(html: str, out_pdf: Path, timeout_s: int = 180) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html, encoding="utf-8")
        cmd = [
            chrome_path(), "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={tmp}/profile",
            "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}", src.as_uri(),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if res.returncode != 0 or not out_pdf.exists() or out_pdf.stat().st_size == 0:
        raise RenderError(f"Chrome failed ({res.returncode}): {res.stderr[-500:]}")


def pdf_page_count(path: Path) -> int:
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        return doc.page_count


def source_page_count(raw: bytes) -> int:
    """Original pages = explicit page breaks in the source HTML + 1."""
    return len(_BREAK_RE.findall(raw.decode("utf-8", errors="replace"))) + 1


def render_fitted(raw: bytes, out_pdf: Path, images: dict[str, bytes] | None = None) -> dict:
    """
    Render at the largest zoom whose page count doesn't exceed the source's.
    Returns {zoom, pages, source_pages, fitted, images_removed[, images_embedded]}.
    """
    target = source_page_count(raw)
    pages, zoom, stats = 0, ZOOM_STEPS[0], {}
    for zoom in ZOOM_STEPS:
        html, stats = sanitize_html(raw, MARGIN_IN, zoom, images)
        render_pdf(html, out_pdf)
        pages = pdf_page_count(out_pdf)
        if pages <= target:
            break
    return {"zoom": zoom, "pages": pages, "source_pages": target,
            "fitted": pages <= target, **stats}
