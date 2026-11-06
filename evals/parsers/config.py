"""
Parser configurations and the parse cache layout (scope v2 §6.0, §9.2).

Stdlib only: imported both by the Docling runner (venv-docling) and by the
evals code in the app venv, which reads the cache without importing Docling.

Cache: data/parse_cache/<config_name>/<doc_id>.json
Each file records the cache key: hash(doc sha256, parser, parser version, config).
A cached file is reused only when its key matches (F5).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO / "data" / "parse_cache"


@dataclass(frozen=True)
class ParserConfig:
    name: str                    # P1 / P2 (scope v2 §9.3)
    parser: str = "docling"
    table_mode: str = "fast"     # TableFormer: fast | accurate
    ocr_engine: str = "tesseract_cli"
    ocr_lang: str = "eng"
    device: str = "mps"          # Apple GPU; cpu as fallback
    num_threads: int = 8

    def as_dict(self) -> dict:
        return asdict(self)


P1 = ParserConfig(name="P1", table_mode="fast")
P2 = ParserConfig(name="P2", table_mode="accurate")
# Apple Vision OCR variants, parsed on the simulated scans only. Decision 2026-11-06
# (S-2026-11-06-1): OCR'd text used DOWNSTREAM (chunks, retrieval, answers) comes from these;
# confidence metrics and H7 keep using the Tesseract run (P1/P2), whose scores are informative.
P1V = ParserConfig(name="P1V", table_mode="fast", ocr_engine="ocrmac", ocr_lang="en-US")
P2V = ParserConfig(name="P2V", table_mode="accurate", ocr_engine="ocrmac", ocr_lang="en-US")
CONFIGS = {c.name: c for c in (P1, P2, P1V, P2V)}
VISION_VARIANT = {"P1": "P1V", "P2": "P2V"}


def downstream_config(table_config: str, part: str) -> str:
    """
    Which parse cache feeds chunking/retrieval for a document. Simulated scans use the
    Apple Vision variant of the chosen table config; digital documents use the config
    itself (their text comes from the PDF text layer, so the OCR engine barely matters).
    """
    if table_config not in VISION_VARIANT:
        raise ValueError(f"downstream table config must be one of {sorted(VISION_VARIANT)}")
    return VISION_VARIANT[table_config] if part == "simulated_scan" else table_config


def cache_key(doc_sha256: str, parser_version: str, config: ParserConfig) -> str:
    payload = json.dumps({"doc": doc_sha256, "parser_version": parser_version, **config.as_dict()},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(config: ParserConfig, doc_id: str, root: Path = CACHE_ROOT) -> Path:
    return root / config.name / f"{doc_id}.json"


def load_cached(config: ParserConfig, doc_id: str, root: Path = CACHE_ROOT) -> dict | None:
    path = cache_path(config, doc_id, root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def is_cached(config: ParserConfig, doc_id: str, key: str, root: Path = CACHE_ROOT) -> bool:
    """True if a cache file exists for this doc AND was made with the same key."""
    path = cache_path(config, doc_id, root)
    if not path.exists():
        return False
    try:
        with path.open() as fh:
            head = fh.read(2048)            # meta is written first; avoid loading 20MB
        return f'"cache_key": "{key}"' in head
    except OSError:
        return False
