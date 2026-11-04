"""
Bootstrap statistics (scope v2 §9.6).

- bootstrap_ci:      CI for a statistic of one sample (resampling units, e.g. pages)
- ratio_ci:          CI for a pooled ratio sum(num)/sum(den) (e.g. cell accuracy pooled over pages)
- paired_diff_ci:    CI for the mean per-unit difference b - a (same units, two configurations)
- holm:              Holm step-down adjustment for several p-values
Seeded, 10,000 resamples, 95% by default. Units with undefined values (None) are dropped.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

N_BOOT = 10_000
ALPHA = 0.05


def _clean(xs: Sequence[float | None]) -> np.ndarray:
    return np.array([x for x in xs if x is not None], dtype=float)


def bootstrap_ci(values: Sequence[float | None], stat: Callable[[np.ndarray], float] = np.mean,
                 n: int = N_BOOT, seed: int = 0, alpha: float = ALPHA) -> dict:
    x = _clean(values)
    if len(x) == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat not in (np.mean, np.median) else \
        (x[idx].mean(axis=1) if stat is np.mean else np.median(x[idx], axis=1))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(stat(x)), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def ratio_ci(nums: Sequence[float], dens: Sequence[float], n: int = N_BOOT, seed: int = 0,
             alpha: float = ALPHA) -> dict:
    num, den = np.asarray(nums, dtype=float), np.asarray(dens, dtype=float)
    keep = den > 0
    num, den = num[keep], den[keep]
    if den.sum() == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(den), size=(n, len(den)))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi), "n": int(len(den))}


def paired_diff_ci(a: Sequence[float | None], b: Sequence[float | None], n: int = N_BOOT, seed: int = 0,
                   alpha: float = ALPHA) -> dict:
    """Mean of (b - a) over units where both are defined, with a bootstrap CI and a two-sided
    bootstrap p-value. `better` is 'b' / 'a' only when the CI excludes 0."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"diff": None, "lo": None, "hi": None, "p": None, "n": 0, "better": None}
    d = np.array([y - x for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    better = "b" if lo > 0 else "a" if hi < 0 else None
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p), "n": len(d), "better": better}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
