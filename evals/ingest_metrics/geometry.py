"""Box overlap and one-to-one matching (scope v2 §8.1: a match needs IoU >= 0.5)."""
from __future__ import annotations

from typing import Callable, Sequence

IOU_MATCH = 0.5
Box = Sequence[float]   # [x0, y0, x1, y1], top-left origin


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def greedy_match(labels: list[dict], preds: list[dict], threshold: float = IOU_MATCH,
                 compatible: Callable[[dict, dict], bool] | None = None) -> list[tuple[int, int, float]]:
    """
    One-to-one matching by highest IoU first. Returns (label_idx, pred_idx, iou) pairs
    with iou >= threshold. `compatible` can restrict pairs (e.g. same class).
    """
    cands = []
    for i, l in enumerate(labels):
        for j, p in enumerate(preds):
            if compatible and not compatible(l, p):
                continue
            v = iou(l["bbox"], p["bbox"])
            if v >= threshold:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    used_l, used_p, out = set(), set(), []
    for v, i, j in cands:
        if i in used_l or j in used_p:
            continue
        used_l.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return sorted(out)


def prf(tp: int, n_pred: int, n_label: int) -> dict:
    """Precision / recall / F1; None where undefined (nothing predicted / nothing labelled)."""
    p = tp / n_pred if n_pred else None
    r = tp / n_label if n_label else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": n_pred, "n_label": n_label}
