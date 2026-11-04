"""
Layout, form-field and header/footer metrics (scope v2 §8.1).

Elements are {id, class, bbox} dicts (golden labels and PageView share the shape).
"""
from __future__ import annotations

from collections import Counter

from evals.ingest_metrics.geometry import greedy_match, prf
from evals.ingest_metrics.normalize import norm_match

HEADINGS = {"title", "section_header"}
FURNITURE = {"page_header", "page_footer", "page_number"}


def class_f1(labels: list[dict], preds: list[dict]) -> dict:
    """Per-class P/R/F1 (a match needs the same class and IoU >= 0.5); counts kept so
    pages can be pooled. macro_f1 averages classes that occur in the labels."""
    classes = {e["class"] for e in labels} | {e["class"] for e in preds}
    per = {}
    for c in sorted(classes):
        lab = [e for e in labels if e["class"] == c]
        pre = [e for e in preds if e["class"] == c]
        per[c] = prf(len(greedy_match(lab, pre)), len(pre), len(lab))
    in_labels = [per[c]["f1"] for c in per if per[c]["n_label"] and per[c]["f1"] is not None]
    return {"per_class": per, "macro_f1": sum(in_labels) / len(in_labels) if in_labels else None}


def section_header_recall(labels: list[dict], preds: list[dict]) -> dict:
    """★ Share of labelled titles/section headers detected as a title or section header."""
    lab = [e for e in labels if e["class"] in HEADINGS]
    pre = [e for e in preds if e["class"] in HEADINGS]
    hit = len(greedy_match(lab, pre))
    return {"hit": hit, "total": len(lab), "recall": hit / len(lab) if lab else None}


def kendall_tau(a: list, b: list) -> float | None:
    """Kendall's tau between two orderings of the same items (no ties). None for < 2 items."""
    pos = {x: i for i, x in enumerate(b)}
    seq = [pos[x] for x in a if x in pos]
    n = len(seq)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def reading_order_tau(lab_elements: list[dict], lab_order: list[str],
                      pred_elements: list[dict], pred_order: list[str]) -> dict:
    """
    Kendall's tau over body elements present in both: labelled elements are matched to
    predicted ones by IoU (any class); headers/footers/page numbers are left out.
    """
    lab = [e for e in lab_elements if e["class"] not in FURNITURE and e["id"] in lab_order]
    pre = [e for e in pred_elements if e["class"] not in FURNITURE]
    pairs = greedy_match(lab, pre)
    lab_to_pred = {lab[i]["id"]: pre[j]["id"] for i, j, _ in pairs}
    in_label_order = [lab_to_pred[x] for x in lab_order if x in lab_to_pred]
    in_pred_order = [x for x in pred_order if x in set(lab_to_pred.values())]
    return {"tau": kendall_tau(in_label_order, in_pred_order), "n": len(in_label_order)}


def header_footer(label_lines: list[str], pred_lines: list[str]) -> dict:
    """Precision/recall of header/footer lines the parser identified, by normalized text (multiset)."""
    lab, pre = Counter(map(norm_match, label_lines)), Counter(map(norm_match, pred_lines))
    lab.pop("", None)
    pre.pop("", None)
    return prf(sum((lab & pre).values()), sum(pre.values()), sum(lab.values()))


def key_values(labels: list[dict], preds: list[dict]) -> dict:
    """Pairs matched on normalized key (multiset); value exact-match among matched keys."""
    lab_by_key: dict[str, list[str]] = {}
    for kv in labels:
        lab_by_key.setdefault(norm_match(kv["key"]), []).append(norm_match(kv.get("value", "")))
    matched = exact = 0
    for kv in preds:
        k = norm_match(kv["key"])
        if lab_by_key.get(k):
            want = lab_by_key[k].pop(0)
            matched += 1
            exact += int(norm_match(kv.get("value", "")) == want)
    out = prf(matched, len(preds), len(labels))
    out["value_exact"] = exact
    out["value_exact_rate"] = exact / matched if matched else None
    return out
