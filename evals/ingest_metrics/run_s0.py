"""
Stage S0 — ingestion and parsing metrics (scope v2 §8.1, §9.3).

    venv/bin/python -m evals.ingest_metrics.run_s0                       # verified labels only
    venv/bin/python -m evals.ingest_metrics.run_s0 --include-unverified  # DRY RUN: plumbing check, not results
    venv/bin/python -m evals.ingest_metrics.run_s0 --skip-corpus         # golden-set part only
    venv/bin/python -m evals.ingest_metrics.run_s0 --mlflow              # also log to local MLflow

Two parts:
1. Golden set (Ingest metrics): the hand-verified pages vs each parser config.
   Decision (D3): P1 vs P2 by cell accuracy, paired bootstrap over labelled tables;
   a tie goes to the faster config.
2. Full corpus (Auto metrics): text accuracy against the PDF text layer (scans: their
   source page), confidence, throughput. Every page, every config.

Outputs (gitignored): data/results/s0/summary.json, golden_pages.jsonl, corpus_pages.jsonl
Pre-filled (unverified) labels ARE Docling P1's output, so a dry run scores P1 near-perfect
by construction. Only verified labels produce results.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np

from evals.corpus.manifest import read_manifest
from evals.golden.ingest.labeling import PAGES as LABEL_DIR
from evals.ingest_metrics import layout as L
from evals.ingest_metrics import tables as T
from evals.ingest_metrics.confidence import document_ingestion_score, page_confidence
from evals.ingest_metrics.text import bag_word_recall, cer, numeric_fidelity, text_layer_usable, wer
from evals.parsers.config import CONFIGS, load_cached
from evals.parsers.docling_doc import all_page_views
from evals.stats import bootstrap_ci, paired_diff_ci, ratio_ci

log = logging.getLogger("evals.s0")
REPO = Path(__file__).resolve().parents[2]
PDF_DIR = REPO / "data" / "corpus" / "pdf"
OUT = REPO / "data" / "results" / "s0"
STARRED = ("cer_median", "numeric_fidelity", "cell_accuracy", "section_header_recall")


# ---------------------------------------------------------------- shared helpers
class Docs:
    """Lazy per-(config, doc) cache of page views, plus PDF text layers."""

    def __init__(self):
        self._views: dict[tuple[str, str], dict] = {}
        self._cached: dict[tuple[str, str], dict | None] = {}
        self._text: dict[tuple[str, int], str] = {}

    def cached(self, cfg: str, doc_id: str) -> dict | None:
        key = (cfg, doc_id)
        if key not in self._cached:
            self._cached[key] = load_cached(CONFIGS[cfg], doc_id)
            if self._cached[key] is not None:
                self._views[key] = all_page_views(self._cached[key]["document"])
        return self._cached[key]

    def view(self, cfg: str, doc_id: str, page: int):
        if self.cached(cfg, doc_id) is None:
            return None
        return self._views[(cfg, doc_id)].get(page)

    def drop(self, cfg: str, doc_id: str) -> None:
        self._cached.pop((cfg, doc_id), None)
        self._views.pop((cfg, doc_id), None)

    def text_layer(self, doc_id: str, page: int) -> str:
        key = (doc_id, page)
        if key not in self._text:
            with fitz.open(PDF_DIR / f"{doc_id}.pdf") as d:
                self._text[key] = d[page - 1].get_text()
        return self._text[key]


def reference_for(rows: dict, doc_id: str, page: int, docs: Docs) -> tuple[str, str]:
    """(reference text, source description). Scans use the page they were rendered from."""
    row = rows[doc_id]
    if row.part == "simulated_scan":
        _, src_doc, src_page = row.role.split(":")
        return docs.text_layer(src_doc, int(src_page.lstrip("p"))), f"{src_doc}:{src_page}"
    return docs.text_layer(doc_id, page), "text_layer"


def text_scores(hyp: str, ref: str) -> dict:
    usable, why = text_layer_usable(ref)
    if not usable:
        return {"ref_usable": False, "excluded_reason": why}
    return {"ref_usable": True, "cer": cer(hyp, ref), "wer": wer(hyp, ref),
            "numeric_fidelity": numeric_fidelity(hyp, ref), "bag_word_recall": bag_word_recall(hyp, ref)}


def _q(vals, q):
    v = [x for x in vals if x is not None]
    return float(np.quantile(v, q)) if v else None


# ---------------------------------------------------------------- part 1: golden set
def load_labels(include_unverified: bool) -> list[dict]:
    labs = [json.loads(p.read_text()) for p in sorted(LABEL_DIR.glob("*.json"))]
    return [l for l in labs if include_unverified or l["status"] == "verified"]


def score_golden_page(lab: dict, view, rows: dict, docs: Docs) -> dict:
    rec = {"page_id": lab["page_id"], "stratum": lab["stratum"], "from_scratch": lab["from_scratch"],
           "status": lab["status"]}
    if view is None:
        return {**rec, "missing_parse": True}
    ref, ref_src = reference_for(rows, lab["doc_id"], lab["page"], docs)
    if lab.get("reference_text_source") == "hand_typed":
        ref, ref_src = lab["reference_text"], "hand_typed"
    rec.update(text_scores(view.text(), ref), ref_source=ref_src)
    tm = T.page_table_metrics(lab["tables"], view.tables)
    rec["tables"] = {"detection": tm["detection"], "cells_correct": tm["cells"]["correct"],
                     "cells_total": tm["cells"]["total"], "per_table": [
                         {**pt, **ca} for pt, ca in zip(tm["per_table"], tm["cells"]["per_table"])]}
    rec["class_f1"] = L.class_f1(lab["layout_elements"], view.elements)["per_class"]
    rec["section_headers"] = L.section_header_recall(lab["layout_elements"], view.elements)
    rec["reading_order"] = L.reading_order_tau(lab["layout_elements"], lab["reading_order"],
                                               view.elements, view.reading_order)
    rec["header_footer"] = L.header_footer(lab["header_footer_lines"], view.header_footer_lines)
    rec["key_values"] = L.key_values(lab["key_values"], view.key_values)
    return rec


def aggregate_golden(recs: list[dict]) -> dict:
    recs = [r for r in recs if not r.get("missing_parse")]
    if not recs:
        return {"pages": 0}
    text = [r for r in recs if r.get("ref_usable")]
    cer_v = [r["cer"] for r in text]
    tables = [t for r in recs for t in r["tables"]["per_table"]]
    det = defaultdict(int)
    for r in recs:
        for k in ("tp", "n_pred", "n_label"):
            det[k] += r["tables"]["detection"][k]
    cls = defaultdict(lambda: defaultdict(int))
    for r in recs:
        for c, v in r["class_f1"].items():
            for k in ("tp", "n_pred", "n_label"):
                cls[c][k] += v[k]
    per_class = {c: L.prf(v["tp"], v["n_pred"], v["n_label"]) for c, v in cls.items()}
    macro = [v["f1"] for v in per_class.values() if v["n_label"] and v["f1"] is not None]
    hf = {k: sum(r["header_footer"][k] for r in recs) for k in ("tp", "n_pred", "n_label")}
    kv = {k: sum(r["key_values"][k] for r in recs) for k in ("tp", "n_pred", "n_label", "value_exact")}
    taus = [r["reading_order"]["tau"] for r in recs]
    hdr = [t["header_rows_correct"] for t in tables if t["header_rows_correct"] is not None]
    return {
        "pages": len(recs),
        "text_pages": len(text),
        "excluded_text_pages": {r["page_id"]: r["excluded_reason"] for r in recs if not r.get("ref_usable")},
        "cer_median": bootstrap_ci(cer_v, stat=np.median),
        "cer_p90": _q(cer_v, 0.9),
        "wer_median": _q([r["wer"] for r in text], 0.5),
        "numeric_fidelity": bootstrap_ci([r["numeric_fidelity"] for r in text]),
        "table_detection": L.prf(det["tp"], det["n_pred"], det["n_label"]),
        "cell_accuracy": ratio_ci([r["tables"]["cells_correct"] for r in recs],
                                  [r["tables"]["cells_total"] for r in recs]),
        "teds_mean": float(np.mean([t["teds"] for t in tables])) if tables else None,
        "header_row_accuracy": sum(hdr) / len(hdr) if hdr else None,
        "key_values": {**L.prf(kv["tp"], kv["n_pred"], kv["n_label"]),
                       "value_exact_rate": kv["value_exact"] / kv["tp"] if kv["tp"] else None},
        "class_f1": per_class,
        "macro_f1": sum(macro) / len(macro) if macro else None,
        "section_header_recall": ratio_ci([r["section_headers"]["hit"] for r in recs],
                                          [r["section_headers"]["total"] for r in recs]),
        "reading_order_tau_mean": _q(taus, 0.5) if False else (float(np.mean([t for t in taus if t is not None]))
                                                                 if any(t is not None for t in taus) else None),
        "header_footer": L.prf(hf["tp"], hf["n_pred"], hf["n_label"]),
    }


def golden(configs: list[str], include_unverified: bool, rows: dict, docs: Docs) -> dict:
    labels = load_labels(include_unverified)
    out = {"labels": len(labels), "verified": sum(l["status"] == "verified" for l in labels), "configs": {}}
    per_cfg_recs: dict[str, list[dict]] = {}
    for cfg in configs:
        recs = [score_golden_page(l, docs.view(cfg, l["doc_id"], l["page"]), rows, docs) for l in labels]
        per_cfg_recs[cfg] = recs
        groups = {"all": recs, "from_scratch": [r for r in recs if r["from_scratch"]],
                  "prefilled": [r for r in recs if not r["from_scratch"]]}
        for s in sorted({r["stratum"] for r in recs}):
            groups[f"stratum:{s}"] = [r for r in recs if r["stratum"] == s]
        out["configs"][cfg] = {g: aggregate_golden(rs) for g, rs in groups.items()}
    out["records"] = per_cfg_recs
    if {"P1", "P2"} <= set(configs):
        out["decision"] = decide(per_cfg_recs["P1"], per_cfg_recs["P2"])
        out["h10"] = h10(per_cfg_recs["P1"], per_cfg_recs["P2"])
    return out


def decide(p1: list[dict], p2: list[dict]) -> dict:
    """D3 for S0: cell accuracy, paired over labelled tables; tie -> faster config."""
    def per_table(recs):
        return {(r["page_id"], i): t["accuracy"] for r in recs if not r.get("missing_parse")
                for i, t in enumerate(r["tables"]["per_table"])}
    a, b = per_table(p1), per_table(p2)
    keys = sorted(set(a) & set(b))
    res = paired_diff_ci([a[k] for k in keys], [b[k] for k in keys])
    winner = {"a": "P1", "b": "P2"}.get(res["better"])
    return {"metric": "cell_accuracy (P2 - P1), paired over labelled tables", **res,
            "winner": winner, "rule": "CI excludes 0" if winner else "tie: pick the faster config"}


def h10(p1: list[dict], p2: list[dict], stratum: str = "table_heavy") -> dict:
    """
    H10: P2 beats P1 on cell accuracy for the table-heavy stratum. Same paired test as the
    decision, restricted to that stratum (rule fixed in docs/20_s0_results.md §2).
    """
    res = decide([r for r in p1 if r["stratum"] == stratum], [r for r in p2 if r["stratum"] == stratum])
    verdict = {"P2": "supported", "P1": "not supported"}.get(res["winner"], "inconclusive")
    return {"stratum": stratum, **res, "verdict": verdict}


# ---------------------------------------------------------------- part 2: full corpus
def corpus(configs: list[str], rows: dict, docs: Docs) -> dict:
    out: dict = {"configs": {}}
    records: list[dict] = []
    for cfg in configs:
        by_part = defaultdict(list)
        ingest_scores, seconds, pages_parsed, missing = [], defaultdict(float), defaultdict(int), []
        for doc_id, row in rows.items():
            cached = docs.cached(cfg, doc_id)
            if cached is None:
                missing.append(doc_id)
                continue
            seconds[row.part] += cached["meta"]["seconds"]
            pages_parsed[row.part] += int(row.pdf_pages or 0)
            ingest_scores.append({"doc_id": doc_id, "part": row.part, **document_ingestion_score(cached)})
            for page in range(1, int(row.pdf_pages or 0) + 1):
                view = docs.view(cfg, doc_id, page)
                ref, ref_src = reference_for(rows, doc_id, page, docs)
                rec = {"config": cfg, "doc_id": doc_id, "part": row.part, "page": page,
                       **text_scores(view.text() if view else "", ref), "ref_source": ref_src,
                       "confidence": page_confidence(cached, page)}
                if row.part == "simulated_scan":
                    rec["scan_level"] = row.renderer.split("level=")[-1]
                by_part[row.part].append(rec)
                records.append(rec)
            docs.drop(cfg, doc_id)           # keep memory flat across 92 documents
        cfg_out = {}
        for part, recs in by_part.items():
            text = [r for r in recs if r.get("ref_usable")]
            ocr = [r["confidence"] for r in recs if r["confidence"]["ocr"]["n"]]
            cfg_out[part] = {
                "pages": len(recs), "text_pages": len(text),
                "excluded": dict(sorted(_count(r["excluded_reason"].split()[0] if r["excluded_reason"].startswith("only") else r["excluded_reason"]
                                               for r in recs if not r.get("ref_usable")).items())),
                "cer_median": bootstrap_ci([r["cer"] for r in text], stat=np.median),
                "cer_p90": _q([r["cer"] for r in text], 0.9),
                "wer_median": _q([r["wer"] for r in text], 0.5),
                "numeric_fidelity": bootstrap_ci([r["numeric_fidelity"] for r in text]),
                "bag_word_recall_mean": float(np.mean([r["bag_word_recall"] for r in text])) if text else None,
                "ocr_pages": len(ocr),
                "ocr_word_conf_mean": float(np.mean([c["ocr"]["mean"] for c in ocr])) if ocr else None,
                "low_conf_rate_90_mean": float(np.mean([c["low_conf_rate_90"] for c in ocr])) if ocr else None,
                "low_conf_rate_80_mean": float(np.mean([c["low_conf_rate_80"] for c in ocr])) if ocr else None,
                "low_confidence_pages": sum(r["confidence"]["low_confidence_page"] for r in recs),
                "pages_per_minute": 60 * pages_parsed[part] / seconds[part] if seconds[part] else None,
            }
        scans = by_part.get("simulated_scan", [])
        if scans:
            cfg_out["simulated_scan_by_level"] = {
                lvl: {"pages": len(rs), "cer_median": _q([r["cer"] for r in rs if r.get("ref_usable")], 0.5),
                      "bag_word_recall_mean": float(np.mean([r["bag_word_recall"] for r in rs if r.get("ref_usable")])),
                      "numeric_fidelity_mean": float(np.mean([r["numeric_fidelity"] for r in rs
                                                              if r.get("ref_usable") and r["numeric_fidelity"] is not None]))}
                for lvl in ("light", "medium", "heavy") for rs in [[r for r in scans if r.get("scan_level") == lvl]] if rs}
        cfg_out["document_ingestion_scores"] = ingest_scores
        cfg_out["missing_parse"] = missing
        out["configs"][cfg] = cfg_out
    out["h6"] = h6(records)
    return out, records


def _count(items):
    c = defaultdict(int)
    for x in items:
        c[x] += 1
    return c


def h6(records: list[dict]) -> dict:
    """
    H6: simulated-scan pages have at least twice the CER of the digital pages they were
    rendered from. Paired per scan, within each config: scan CER vs its source page's CER.
    """
    digital = {(r["config"], r["doc_id"], r["page"]): r.get("cer") for r in records
               if r["part"] != "simulated_scan" and r.get("ref_usable")}
    res = {}
    for cfg in sorted({r["config"] for r in records}):
        pairs = []
        for r in records:
            if r["config"] != cfg or r["part"] != "simulated_scan" or not r.get("ref_usable"):
                continue
            src_doc, src_page = r["ref_source"].split(":")
            src = digital.get((cfg, src_doc, int(src_page.lstrip("p"))))
            if src is not None:
                pairs.append((r["cer"], src))
        if not pairs:
            continue
        scan_med = float(np.median([a for a, _ in pairs]))
        src_med = float(np.median([b for _, b in pairs]))
        res[cfg] = {"pairs": len(pairs), "scan_cer_median": scan_med, "source_cer_median": src_med,
                    "ratio": scan_med / src_med if src_med else None,
                    "supported": scan_med >= 2 * src_med,
                    "note": "CER is measured against text-layer order, so reading-order differences "
                            "raise CER on digital pages too; read with bag_word_recall (order-free)"}
    return res


# ---------------------------------------------------------------- output
def flat_metrics(summary: dict) -> dict[str, float]:
    """Flatten headline numbers for MLflow (names map 1:1 onto §8.1)."""
    out = {}
    for cfg, groups in summary.get("golden", {}).get("configs", {}).items():
        g = groups.get("all", {})
        for k in STARRED:
            v = g.get(k)
            if isinstance(v, dict) and v.get("point") is not None:
                out[f"{cfg}.golden.{k}"] = v["point"]
                out[f"{cfg}.golden.{k}_lo"], out[f"{cfg}.golden.{k}_hi"] = v["lo"], v["hi"]
        for k in ("teds_mean", "macro_f1", "reading_order_tau_mean", "header_row_accuracy"):
            if g.get(k) is not None:
                out[f"{cfg}.golden.{k}"] = g[k]
    for cfg, parts in summary.get("corpus", {}).get("configs", {}).items():
        for part in ("edgar", "consumer", "simulated_scan"):
            p = parts.get(part)
            if not p:
                continue
            if p["cer_median"]["point"] is not None:
                out[f"{cfg}.corpus.{part}.cer_median"] = p["cer_median"]["point"]
            if p["numeric_fidelity"]["point"] is not None:
                out[f"{cfg}.corpus.{part}.numeric_fidelity"] = p["numeric_fidelity"]["point"]
            if p["pages_per_minute"]:
                out[f"{cfg}.corpus.{part}.pages_per_minute"] = p["pages_per_minute"]
    return out


def log_mlflow(summary: dict) -> None:
    import mlflow
    store = REPO / "data" / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    exp = "loandoc_ingestion_evals"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name="S0" + (" DRY RUN" if summary["dry_run"] else "")):
        mlflow.set_tags({"stage": "S0", "dry_run": str(summary["dry_run"])})
        mlflow.log_params({"configs": ",".join(summary["configs"]),
                           "labels": summary.get("golden", {}).get("labels"),
                           "verified": summary.get("golden", {}).get("verified")})
        mlflow.log_metrics(flat_metrics(summary))
        mlflow.log_artifact(str(OUT / "summary.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["P1", "P2"], choices=sorted(CONFIGS))
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument("--skip-golden", action="store_true")
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = {r.doc_id: r for r in read_manifest()}
    docs = Docs()
    summary = {"stage": "S0", "configs": args.configs, "dry_run": args.include_unverified}
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_golden:
        g = golden(args.configs, args.include_unverified, rows, docs)
        records = g.pop("records")
        with (OUT / "golden_pages.jsonl").open("w") as fh:
            for cfg, recs in records.items():
                for r in recs:
                    fh.write(json.dumps({"config": cfg, **r}) + "\n")
        summary["golden"] = g
        log.info("golden: %d label files (%d verified)", g["labels"], g["verified"])
    if not args.skip_corpus:
        c, records = corpus(args.configs, rows, docs)
        with (OUT / "corpus_pages.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        summary["corpus"] = c

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if args.mlflow:
        log_mlflow(summary)
    print(json.dumps(flat_metrics(summary), indent=2))
    if summary["dry_run"]:
        print("\nDRY RUN: unverified labels are Docling P1's own output. These are not results.")
    if "decision" in summary.get("golden", {}):
        print("\nS0 decision:", json.dumps(summary["golden"]["decision"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
