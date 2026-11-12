"""
S0 part 3: production cleaner + validator on Docling output, full corpus (scope v2 §8.1).

    venv/bin/python -m evals.ingest_metrics.run_cleaning                  # P2 + P2V, cleaner v1 and v2
    venv/bin/python -m evals.ingest_metrics.run_cleaning --configs P1 P2 --versions v2

Output (gitignored): data/results/s0/cleaning.json, per-document records + per (config, part) aggregates.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import structlog

from evals.corpus.manifest import read_manifest
from evals.ingest_metrics.cleaning import aggregate, run_doc
from evals.ingest_metrics.run_s0 import Docs, reference_for
from evals.ingest_metrics.text import text_layer_usable
from evals.parsers.config import CONFIGS, load_cached
from evals.parsers.production_adapter import to_extraction_result

from app.pipeline.cleaner import CLEANING_VERSIONS  # noqa: E402  (path set by production_adapter)

OUT = Path("data/results/s0/cleaning.json")
log = logging.getLogger("run_cleaning")


def _quiet_production_logs() -> None:
    """The production modules log every stage at INFO; 92 docs x configs is a lot of noise."""
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def run(configs: list[str], versions: list[str]) -> dict:
    rows = {r.doc_id: r for r in read_manifest()}
    docs = Docs()                       # only used for reference text layers
    out: dict = {"configs": {}, "documents": []}
    for cfg, version in [(c, v) for c in configs for v in versions]:
        by_part = defaultdict(list)
        missing = []
        for doc_id, row in rows.items():
            cached = load_cached(CONFIGS[cfg], doc_id)
            if cached is None:
                missing.append(doc_id)
                continue

            def reference(page: int, _d=doc_id) -> str | None:
                ref, _ = reference_for(rows, _d, page, docs)
                return ref if text_layer_usable(ref)[0] else None

            rec = {"config": cfg, "doc_id": doc_id, "part": row.part,
                   **run_doc(to_extraction_result(cached, doc_id), reference, version)}
            by_part[row.part].append(rec)
            out["documents"].append(rec)
        key = f"{cfg}/{version}"
        out["configs"][key] = {part: aggregate(recs) for part, recs in by_part.items()}
        out["configs"][key]["missing_parse"] = len(missing)
        log.info("%s: %d docs, %d missing", key, sum(len(v) for v in by_part.values()), len(missing))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["P2", "P2V"], choices=sorted(CONFIGS))
    ap.add_argument("--versions", nargs="+", default=list(CLEANING_VERSIONS), choices=CLEANING_VERSIONS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _quiet_production_logs()
    res = run(args.configs, args.versions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["configs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
