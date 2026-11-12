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
- Chrome path from CHROME_PATH, defaulting to the macOS install
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MARGIN_IN = 0.5
ZOOM = 1.0
ZOOM_STEPS = (1.0, 0.9, 0.8, 0.7)
_BREAK_RE = re.compile(r"(?:page-)?break-(?:before|after)\s*:\s*(?:always|page)", re.IGNORECASE)


def print_head(margin_in: float = MARGIN_IN, zoom: float = ZOOM) -> str:
    return (
        '<meta charset="utf-8">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        f"<style>@page {{ size: Letter; margin: {margin_in}in; }} body {{ margin: 0; zoom: {zoom}; }}</style>"
    )


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


def sanitize_html(raw: bytes, margin_in: float = MARGIN_IN, zoom: float = ZOOM) -> tuple[str, dict]:
    """Strip scripts and images, add print settings. Returns (html, stats)."""
    html = raw.decode("utf-8", errors="replace")
    stats = {
        "images_removed": len(re.findall(r"<img\b", html, re.I)),
        "scripts_removed": len(re.findall(r"<script\b", html, re.I)),
    }
    html = re.sub(r"<script\b.*?</script\s*>", "", html, flags=re.I | re.S)
    html = re.sub(r"<img\b[^>]*>", "", html, flags=re.I)
    if re.search(r"<head\b[^>]*>", html, re.I):
        html = re.sub(r"(<head\b[^>]*>)", lambda m: m.group(1) + print_head(margin_in, zoom),
                      html, count=1, flags=re.I)
    else:
        html = print_head(margin_in, zoom) + html
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


def render_fitted(raw: bytes, out_pdf: Path) -> dict:
    """
    Render at the largest zoom whose page count doesn't exceed the source's.
    Returns {zoom, pages, source_pages, fitted, images_removed}.
    """
    target = source_page_count(raw)
    pages, zoom, stats = 0, ZOOM_STEPS[0], {}
    for zoom in ZOOM_STEPS:
        html, stats = sanitize_html(raw, MARGIN_IN, zoom)
        render_pdf(html, out_pdf)
        pages = pdf_page_count(out_pdf)
        if pages <= target:
            break
    return {"zoom": zoom, "pages": pages, "source_pages": target,
            "fitted": pages <= target, **stats}
