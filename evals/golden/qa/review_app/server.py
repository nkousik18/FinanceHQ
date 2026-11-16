"""
Local review app for the QA drafts (R3 step 3).

    venv/bin/python -m evals.golden.qa.review_app.server        # then open http://127.0.0.1:8766
    venv/bin/python -m evals.golden.qa.review_app.server --port 9001 --name kousik

Standard library only; binds to 127.0.0.1. Reads drafts/candidates.jsonl, writes drafts/reviews.json
(see store.py). Page images are rendered from data/corpus/pdf on demand (cache: data/qa_review_images/).

API
  GET  /                                 the app (index.html)
  GET  /api/meta                         reviewer, drop reasons
  GET  /api/list                         candidates with status + kept-per-type counts
  GET  /api/cand/<cid>                   candidate, its review, shown pages with span highlights
  GET  /api/page/<doc_id>/<page>         one page's text (prev/next browsing)
  POST /api/check/<cid>   body = final   -> {"errors": [...]}   (no write)
  PUT  /api/review/<cid>  body = {decision, final, drop_reason, note} -> saves; "keep" refused with errors
  GET  /img/<doc_id>/<page>.png          page image
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evals.golden.qa.review_app.store import DROP_REASONS, ReviewStore

log = logging.getLogger("qa_review_app")

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parents[3]
DRAFTS = APP_DIR.parent / "drafts"
MAX_BODY = 2_000_000
PAGE_RE = re.compile(r"^/api/page/([A-Za-z0-9_]+)/(\d+)$")
IMG_RE = re.compile(r"^/img/([A-Za-z0-9_]+)/(\d+)\.png$")


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QAReview/1"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            p = self.path
            try:
                if p in ("/", "/index.html"):
                    self._send(200, (APP_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif p == "/api/meta":
                    self._json(200, {"reviewer": store.reviewer, "drop_reasons": list(DROP_REASONS)})
                elif p == "/api/list":
                    self._json(200, store.list())
                elif p.startswith("/api/cand/"):
                    self._json(200, store.get(p.rsplit("/", 1)[1]))
                elif m := PAGE_RE.match(p):
                    self._json(200, store.page_view(m[1], int(m[2])))
                elif m := IMG_RE.match(p):
                    self._send(200, store.image(m[1], int(m[2])), "image/png")
                else:
                    self._json(404, {"error": "not found"})
            except (KeyError, FileNotFoundError):
                self._json(404, {"error": "unknown candidate, document or page"})

        def do_POST(self):
            if not self.path.startswith("/api/check/"):
                return self._json(404, {"error": "not found"})
            try:
                self._json(200, {"errors": store.check(self.path.rsplit("/", 1)[1], self._body())})
            except KeyError:
                self._json(404, {"error": "unknown candidate"})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

        def do_PUT(self):
            if not self.path.startswith("/api/review/"):
                return self._json(404, {"error": "not found"})
            try:
                ok, errors = store.save(self.path.rsplit("/", 1)[1], self._body())
            except KeyError:
                return self._json(404, {"error": "unknown candidate"})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(200 if ok else HTTPStatus.UNPROCESSABLE_ENTITY, {"saved": ok, "errors": errors})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--name", default="kousik", help="reviewer name recorded on each review")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    from evals.golden.qa.validate import CacheText, corpus_docs
    docs = corpus_docs()
    store = ReviewStore(DRAFTS / "candidates.jsonl", DRAFTS / "reviews.json", docs, CacheText(docs), args.name,
                        pdf_dir=REPO / "data" / "corpus" / "pdf", image_dir=REPO / "data" / "qa_review_images")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"QA review app: http://127.0.0.1:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
