"""
Blind labelling app for judge calibration (scope v2 §8.6, R6 step 3).

    venv/bin/python -m evals.generation.calibration_app.server            # http://127.0.0.1:8767
    venv/bin/python -m evals.generation.calibration_app.server --n 50     # after extending to 50

Shows, per sampled answer: question, reference answer + key points, the answer, and the exact context
the generator saw (cited chunks marked). Hidden: the judge's grade, every metric, and which
configuration produced the answer. Labels go to evals/golden/judge_calibration/labels.json:
correctness (correct/partial/wrong), faithful (every claim supported by the context), relevance 1-5,
optional note. Standard library only; binds to 127.0.0.1.

API
  GET /                 the app          GET /api/list        items + labelled flag
  GET /api/item/<i>     one item         PUT /api/label/<i>   {correctness, faithful, relevance, note}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evals.generation import calibration as CAL
from evals.generation import metrics as GM
from evals.generation import run as GR

APP_DIR = Path(__file__).resolve().parent
log = logging.getLogger("calibration_app")


class LabelStore:
    def __init__(self, keys: list[tuple[str, str]], questions: dict, cfg, store, labels_path: Path,
                 labeler: str, out_root: Path = GR.OUT_ROOT, retrieval_root: Path = GR.RETRIEVAL_ROOT):
        self.keys, self.questions, self.labels_path, self.labeler = keys, questions, labels_path, labeler
        runs = {r.name: r for r in cfg.runs}
        self._answers, self._retr = {}, {}
        for name in {k[0] for k in keys}:
            r = runs[name]
            self._answers[name] = {a["qid"]: a for a in GR.read_jsonl(GR.out_dir(cfg, r, "dev", out_root) / "answers.jsonl")}
            rdir = GR.retrieval_dir(r, "dev", retrieval_root)
            self._retr[name] = (json.loads((rdir / "summary.json").read_text()),
                                {x["qid"]: x for x in GR.read_jsonl(rdir / "questions.jsonl")})
        self.store = store
        self.data = json.loads(labels_path.read_text()) if labels_path.exists() else {"labels": {}}

    def key(self, i: int) -> str:
        run, qid = self.keys[i]
        return f"{run}|{qid}"

    def list(self) -> list[dict]:
        return [{"i": i, "qid": q, "type": self.questions[q]["type"], "labelled": self.key(i) in self.data["labels"]}
                for i, (_, q) in enumerate(self.keys)]

    def item(self, i: int) -> dict:
        run, qid = self.keys[i]
        q, a = self.questions[qid], self._answers[run][qid]
        rsum, rows = self._retr[run]
        ctx = GR.context_for(rows[qid], rsum, self.store)
        cited = set(GM.citations(a["answer"]))
        return {"i": i, "n": len(self.keys), "qid": qid, "type": q["type"], "question": q["question"],
                "reference": q["reference_answer"], "key_points": q.get("key_points") or [],
                "answer": a["answer"],
                "context": [{"sid": c.local_id, "doc_id": c.doc_id, "page": c.page, "text": c.text,
                             "cited": c.local_id in cited} for c in ctx],
                "label": self.data["labels"].get(self.key(i))}

    def save(self, i: int, body: dict) -> dict:
        if body.get("correctness") not in CAL.CORRECTNESS:
            raise ValueError("correctness must be correct/partial/wrong")
        if not isinstance(body.get("faithful"), bool):
            raise ValueError("faithful must be true/false")
        rel = body.get("relevance")
        if not isinstance(rel, int) or isinstance(rel, bool) or not 1 <= rel <= 5:
            raise ValueError("relevance must be 1-5")
        lab = {"correctness": body["correctness"], "faithful": body["faithful"], "relevance": rel,
               "note": str(body.get("note") or "")[:2000], "labeled_by": self.labeler,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.data["labels"][self.key(i)] = lab
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.labels_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.labels_path)
        return lab


def make_handler(ls: LabelStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "JudgeCal/1"

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, status, body: bytes, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _index(self) -> int | None:
            try:
                i = int(self.path.rsplit("/", 1)[1])
            except ValueError:
                return None
            return i if 0 <= i < len(ls.keys) else None

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (APP_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/list":
                return self._json(200, ls.list())
            if self.path.startswith("/api/item/") and (i := self._index()) is not None:
                return self._json(200, ls.item(i))
            self._json(404, {"error": "not found"})

        def do_PUT(self):
            if not self.path.startswith("/api/label/") or (i := self._index()) is None:
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 100_000)) or b"{}")
                self._json(200, {"saved": ls.save(i, body)})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--name", default="kousik")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from evals.golden.qa.validate import QUESTIONS, read_questions
    cfg = GR.load_config(CAL.REPO / "evals" / "configs" / "s4_generation.yaml")
    questions = {q["qid"]: q for q in read_questions(QUESTIONS) if q["split"] == "dev"}
    ls = LabelStore(CAL.sample_keys(a.n), questions, cfg, GR.ChunkStore("P2"), CAL.LABELS_FILE, a.name)
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(ls))
    print(f"Judge calibration app: http://127.0.0.1:{a.port}   ({a.n} answers; Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
