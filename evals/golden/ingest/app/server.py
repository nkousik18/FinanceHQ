"""
Local labelling app for the ingestion golden set.

    venv/bin/python -m evals.golden.ingest.app.server            # then open http://127.0.0.1:8765
    venv/bin/python -m evals.golden.ingest.app.server --port 9000 --name kousik

Standard library only. Binds to 127.0.0.1 (this machine only). Reads and writes
the same files as the CLI: evals/golden/ingest/pages/<page_id>.json.

API
  GET  /                      the app (index.html)
  GET  /api/meta              classes, strata, labeller name
  GET  /api/pages             page list with status and problem counts
  GET  /api/labels/<page_id>  label JSON
  POST /api/validate          body = labels JSON -> {"errors": [...]}  (no write)
  PUT  /api/labels/<page_id>  body = labels JSON -> saves; refuses "verified" with errors
  GET  /img/<page_id>.png     page image
  GET  /api/compare/<page_id> P1 vs P2 tables matched to each label table (adjudication; compare.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evals.golden.ingest import labeling
from evals.golden.ingest.app.compare import compare_page, review_state
from evals.golden.ingest.schema import CLASSES, STRATA, validate_labels

log = logging.getLogger("evals.golden.ingest.app")

APP_DIR = Path(__file__).resolve().parent
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
MAX_BODY = 5_000_000


class LabelStore:
    """File access for the app; kept separate from HTTP so it can be unit-tested."""

    def __init__(self, pages_dir: Path, images_dir: Path, selection: Path, labeller: str, views=None):
        """`views(doc_id, page) -> {"P1": PageView|None, "P2": ...}` enables P1/P2 comparison."""
        self.pages_dir, self.images_dir, self.selection, self.labeller = pages_dir, images_dir, selection, labeller
        self.views = views

    def page_ids(self) -> list[str]:
        return [p["page_id"] for p in json.loads(self.selection.read_text())["pages"]]

    def _check_id(self, page_id: str) -> None:
        if not PAGE_ID_RE.match(page_id) or page_id not in self.page_ids():
            raise KeyError(page_id)

    def list_pages(self) -> list[dict]:
        out = []
        for s in json.loads(self.selection.read_text())["pages"]:
            path = self.pages_dir / f"{s['page_id']}.json"
            lab = json.loads(path.read_text()) if path.exists() else None
            out.append({
                "page_id": s["page_id"], "stratum": s["stratum"], "from_scratch": s["from_scratch"],
                "status": lab["status"] if lab else "missing",
                "problems": len(validate_labels(lab)) if lab else None,
                "elements": len(lab["layout_elements"]) if lab else 0,
                "review": review_state(self._compare(lab)) if lab and self.views else None,
            })
        # from-scratch pages first (label them before seeing Docling output), then by stratum
        out.sort(key=lambda p: (not p["from_scratch"], p["stratum"], p["page_id"]))
        return out

    def load(self, page_id: str) -> dict:
        self._check_id(page_id)
        return json.loads((self.pages_dir / f"{page_id}.json").read_text())

    def save(self, page_id: str, labels: dict) -> tuple[bool, list[str]]:
        """Save labels. Drafts with problems are saved; 'verified' with problems is refused."""
        self._check_id(page_id)
        if labels.get("page_id") != page_id:
            return False, [f"page_id in body ({labels.get('page_id')}) does not match URL ({page_id})"]
        if labels.get("status") == "verified":
            labels.setdefault("labeled_by", self.labeller)
            labels["labeled_by"] = labels["labeled_by"] or self.labeller
            labels["verified_on"] = labels.get("verified_on") or date.today().isoformat()
        errors = validate_labels(labels)
        if errors and labels.get("status") == "verified":
            return False, errors
        path = self.pages_dir / f"{page_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
        tmp.replace(path)
        return True, errors

    def _compare(self, lab: dict) -> dict:
        return compare_page(lab, self.views(lab["doc_id"], lab["page"]))

    def compare(self, page_id: str) -> dict:
        if self.views is None:
            raise KeyError("comparison not enabled")
        return self._compare(self.load(page_id))

    def image_path(self, page_id: str) -> Path:
        self._check_id(page_id)
        return self.images_dir / f"{page_id}.png"


def make_handler(store: LabelStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LabelApp/1"

        def log_message(self, fmt, *args):            # quieter than the default
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self._send(200, (APP_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/api/meta":
                    self._json(200, {"classes": sorted(CLASSES), "strata": sorted(STRATA),
                                     "labeller": store.labeller, "image_dpi": labeling.IMAGE_DPI})
                elif self.path == "/api/pages":
                    self._json(200, store.list_pages())
                elif self.path.startswith("/api/labels/"):
                    self._json(200, store.load(self.path.rsplit("/", 1)[1]))
                elif self.path.startswith("/api/compare/"):
                    self._json(200, store.compare(self.path.rsplit("/", 1)[1]))
                elif self.path.startswith("/img/") and self.path.endswith(".png"):
                    pid = self.path[len("/img/"):-len(".png")]
                    self._send(200, store.image_path(pid).read_bytes(), "image/png")
                else:
                    self._json(404, {"error": "not found"})
            except (KeyError, FileNotFoundError):
                self._json(404, {"error": "unknown page"})

        def do_POST(self):
            if self.path != "/api/validate":
                return self._json(404, {"error": "not found"})
            try:
                self._json(200, {"errors": validate_labels(self._body())})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

        def do_PUT(self):
            if not self.path.startswith("/api/labels/"):
                return self._json(404, {"error": "not found"})
            pid = self.path.rsplit("/", 1)[1]
            try:
                ok, errors = store.save(pid, self._body())
            except KeyError:
                return self._json(404, {"error": "unknown page"})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(200 if ok else HTTPStatus.UNPROCESSABLE_ENTITY, {"saved": ok, "errors": errors})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--name", default="kousik", help="labeller name recorded on verified pages")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from evals.ingest_metrics.run_s0 import Docs   # parse caches; only needed for the P1/P2 comparison
    docs = Docs()
    views = lambda doc_id, page: {cfg: docs.view(cfg, doc_id, page) for cfg in ("P1", "P2")}
    store = LabelStore(labeling.PAGES, labeling.IMAGES, labeling.SELECTION, args.name, views=views)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Labelling app: http://127.0.0.1:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
