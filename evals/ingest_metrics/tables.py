"""
Table metrics (scope v2 §8.1): detection P/R/F1, cell accuracy (★), TEDS, header-row accuracy.

Tables are dicts in the golden-label format: {bbox, num_rows, num_cols, header_rows,
cells: [{row, col, row_span, col_span, text}]}. Parser tables come from
evals/parsers/docling_doc.PageView in the same format.

Split-table rate (tables spanning pages left as unlinked fragments) is NOT computed
here: per-page labels can't say a table continues on the next page. Recorded as a
known gap in the Decision Log.
"""
from __future__ import annotations

from apted import APTED, Config

from evals.ingest_metrics.geometry import greedy_match, prf
from evals.ingest_metrics.normalize import norm_match


def match_tables(labels: list[dict], preds: list[dict]) -> list[tuple[int, int, float]]:
    return greedy_match(labels, preds)


def detection(labels: list[dict], preds: list[dict]) -> dict:
    return prf(len(match_tables(labels, preds)), len(preds), len(labels))


def _cell_at(table: dict, r: int, c: int) -> dict | None:
    for cell in table["cells"]:
        if cell["row"] <= r < cell["row"] + cell["row_span"] and cell["col"] <= c < cell["col"] + cell["col_span"]:
            return cell
    return None


def cell_accuracy_counts(label: dict, pred: dict | None) -> tuple[int, int]:
    """
    (correct, total) over the label's non-empty cells. A cell is correct when its
    normalized text appears in the predicted cell covering the same (row, col).
    Unmatched tables (pred=None) score 0 for every cell. Empty labelled cells are
    skipped: an empty string "appears" in anything.
    """
    correct = total = 0
    for cell in label["cells"]:
        want = norm_match(cell["text"])
        if not want:
            continue
        total += 1
        if pred is None:
            continue
        got = _cell_at(pred, cell["row"], cell["col"])
        if got is not None and want in norm_match(got["text"]):
            correct += 1
    return correct, total


def cell_accuracy(labels: list[dict], preds: list[dict]) -> dict:
    """Pooled over all labelled tables on the page; per-table values for paired comparisons."""
    pairs = {i: j for i, j, _ in match_tables(labels, preds)}
    per_table, correct, total = [], 0, 0
    for i, lab in enumerate(labels):
        c, t = cell_accuracy_counts(lab, preds[pairs[i]] if i in pairs else None)
        correct += c
        total += t
        per_table.append({"label_idx": i, "matched": i in pairs, "correct": c, "total": t,
                          "accuracy": c / t if t else None})
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None,
            "per_table": per_table}


# ---------------------------------------------------------------- TEDS
class _Node:
    __slots__ = ("tag", "colspan", "rowspan", "text", "children")

    def __init__(self, tag, colspan=1, rowspan=1, text="", children=None):
        self.tag, self.colspan, self.rowspan, self.text = tag, colspan, rowspan, text
        self.children = children or []


class _TedsConfig(Config):
    """Standard TEDS costs (Zhong et al., PubTabNet): structure mismatch = 1;
    same-shaped cells cost their normalized text edit distance."""

    def rename(self, a, b):
        if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if a.tag == "td":
            from rapidfuzz.distance import Levenshtein
            return Levenshtein.normalized_distance(a.text, b.text)
        return 0.0

    def children(self, node):
        return node.children


def _tree(table: dict) -> tuple[_Node, int]:
    rows = []
    for r in range(table["num_rows"]):
        tds = [_Node("td", c["col_span"], c["row_span"], norm_match(c["text"]))
               for c in sorted(table["cells"], key=lambda c: c["col"]) if c["row"] == r]
        rows.append(_Node("tr", children=tds))
    root = _Node("table", children=rows)
    size = 1 + len(rows) + sum(len(r.children) for r in rows)
    return root, size


def teds(label: dict, pred: dict | None) -> float:
    """1 - tree edit distance / max(tree sizes). 0 for a missed table."""
    if pred is None:
        return 0.0
    a, na = _tree(label)
    b, nb = _tree(pred)
    dist = APTED(a, b, _TedsConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / max(na, nb))


def header_rows_correct(label: dict, pred: dict | None) -> bool | None:
    """Did the parser mark exactly the labelled header rows? None if the label has none and
    nothing was predicted (no evidence either way)."""
    want = set(label.get("header_rows", []))
    got = set(pred.get("header_rows", [])) if pred else set()
    if not want and not got:
        return None
    return want == got


def page_table_metrics(labels: list[dict], preds: list[dict]) -> dict:
    pairs = {i: j for i, j, _ in match_tables(labels, preds)}
    per = []
    for i, lab in enumerate(labels):
        pred = preds[pairs[i]] if i in pairs else None
        per.append({"teds": teds(lab, pred), "header_rows_correct": header_rows_correct(lab, pred)})
    return {"detection": detection(labels, preds), "cells": cell_accuracy(labels, preds), "per_table": per}
