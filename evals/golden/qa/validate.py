"""
Validate the QA golden set against the corpus (scope v2 F3).

    venv/bin/python -m evals.golden.qa.validate                  # F3 checks
    venv/bin/python -m evals.golden.qa.validate --final          # + §5.1 quotas (complete set)
    venv/bin/python -m evals.golden.qa.validate --file other.jsonl

Page text: parse cache downstream_config("P2", part), cleaner v2 — the text chunkers index.
Exit 1 on any error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import structlog

from evals.golden.qa.schema import DocInfo, mix_report, validate_set
from evals.corpus.manifest import read_manifest

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
TABLE_CONFIG = "P2"
log = logging.getLogger("qa_validate")


def corpus_docs() -> dict[str, DocInfo]:
    return {r.doc_id: DocInfo(r.doc_id, r.part, r.deal_id, r.role) for r in read_manifest()}


class CacheText:
    """Cleaned page text from the parse cache (same pipeline as the chunkers)."""

    def __init__(self, docs: dict[str, DocInfo], table_config: str = TABLE_CONFIG):
        self.docs, self.table_config = docs, table_config

    @lru_cache(maxsize=None)
    def _pages(self, doc_id: str) -> dict[int, str]:
        from evals.chunking.source import build_source
        from evals.parsers.config import CONFIGS, downstream_config, load_cached
        from evals.parsers.production_adapter import to_extraction_result
        info = self.docs.get(doc_id)
        if info is None:
            return {}
        cached = load_cached(CONFIGS[downstream_config(self.table_config, info.part)], doc_id)
        if cached is None:
            log.warning("%s: no parse cache", doc_id)
            return {}
        return build_source(to_extraction_result(cached, doc_id)).pages

    def page(self, doc_id: str, page: int) -> str | None:
        return self._pages(doc_id).get(page)

    def pages(self, doc_id: str) -> list[str]:
        return list(self._pages(doc_id).values())


def read_questions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=QUESTIONS)
    ap.add_argument("--final", action="store_true", help="also enforce the §5.1 quotas")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    docs = corpus_docs()
    qs = read_questions(args.file)
    errs = validate_set(qs, docs, CacheText(docs), final=args.final)
    print(json.dumps({"file": str(args.file), "mix": mix_report(qs, docs), "errors": len(errs),
                      "first_errors": errs[:30]}, indent=1))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
