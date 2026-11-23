"""
FAISS index types for the S3 index study (scope v2 §7 "Index types"): exact `IndexFlatIP` (production)
vs `IndexHNSWFlat` and `IndexIVFFlat`, all inner product on L2-normalized vectors (= cosine).

Index spec (a run config's `index`, D10 parameters, S-2026-11-23-9):
    {type: flat}
    {type: hnsw, M: 32, ef_construction: 200, ef_search: 64}
    {type: ivf, nlist: sqrt, nprobe: 4}        # nlist "sqrt" = round(sqrt(N)), min 1, max N

IVF trains k-means on the index's own vectors (faiss default seed). FAISS runs single-threaded in the
evals (dense._faiss: OpenMP clash with torch), which also makes HNSW graphs and IVF centroids
reproducible.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {"flat": {}, "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 64},
            "ivf": {"nlist": "sqrt", "nprobe": 4}}


def parse_spec(spec: dict | None) -> dict:
    """Validated spec with defaults filled in; None -> flat."""
    spec = dict(spec or {"type": "flat"})
    kind = spec.pop("type", None)
    if kind not in DEFAULTS:
        raise ValueError(f"index type must be one of {sorted(DEFAULTS)}, got {kind!r}")
    unknown = set(spec) - set(DEFAULTS[kind])
    if unknown:
        raise ValueError(f"{kind}: unknown index params {sorted(unknown)}")
    out = {"type": kind, **DEFAULTS[kind], **spec}
    for k, v in out.items():
        if k != "type" and not (v == "sqrt" and k == "nlist") and not (isinstance(v, int) and v >= 1):
            raise ValueError(f"{kind}: {k} must be a positive int")
    return out


def nlist_for(n: int, nlist) -> int:
    k = round(math.sqrt(n)) if nlist == "sqrt" else int(nlist)
    return max(1, min(k, n))


def build(x: np.ndarray, spec: dict | None):
    """A FAISS index over x (float32, normalized) for the spec, search params set."""
    from evals.retrievers.dense import _faiss
    faiss = _faiss()
    s = parse_spec(spec)
    d = x.shape[1]
    if s["type"] == "flat":
        index = faiss.IndexFlatIP(d)
    elif s["type"] == "hnsw":
        index = faiss.IndexHNSWFlat(d, s["M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = s["ef_construction"]
        index.hnsw.efSearch = s["ef_search"]
    else:
        nlist = nlist_for(len(x), s["nlist"])
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.cp.min_points_per_centroid = 1              # small sessions: no "too few points" warnings
        if len(x):
            index.train(x)
        index.nprobe = min(s["nprobe"], nlist)
        index._quantizer = quantizer                      # keep the quantizer alive (SWIG ownership)
    if len(x):
        index.add(x)
    return index


def overlap(approx: list[int], exact: list[int]) -> float | None:
    """§8.3 recall vs exact search: share of exact search's top k that the approximate top k contains."""
    return None if not exact else len(set(approx) & set(exact)) / len(exact)
