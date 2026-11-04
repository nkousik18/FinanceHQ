# Ingestion golden set — labelling guide

30 hand-checked pages used to measure how well the parser reads text, tables,
form fields and layout (scope v2 §5.4, §8.1). About 6 hours in total.

## Files

| Path | What |
|---|---|
| `selection.json` | The 30 pages, their stratum, and which 5 are labelled from scratch |
| `pages/<page_id>.json` | **The labels you edit** (schema in `schema.py`) |
| `images/<page_id>.png` | Clean page image (100 dpi) |
| `data/golden_overlays/<page_id>.png` | Page + 50-point grid + your current boxes (regenerate after edits) |

## Order of work

1. **From-scratch pages first** (`"from_scratch": true`, 5 pages). Their files start empty.
   Label them *without* opening any Docling output, so we can measure how much
   pre-filling biases the labels (§5.4 step 3).
2. Then the 25 pre-filled pages: correct what Docling got wrong, add what it missed.

For each page:

```bash
venv/bin/python -m evals.golden.ingest.labeling render <page_id>   # refresh the overlay
open data/golden_overlays/<page_id>.png                            # look at page + boxes
# edit evals/golden/ingest/pages/<page_id>.json in any editor
venv/bin/python -m evals.golden.ingest.labeling validate           # must pass
```

When a page is done, set `"status": "verified"`, `"labeled_by": "kousik"`, `"verified_on": "YYYY-MM-DD"`.
`labeling status` shows progress.

## Coordinates

PDF points, origin **top-left**, `bbox = [x0, y0, x1, y1]`. A US Letter page is 612 × 792.
Read positions off the grid in the overlay (lines every 50 points, labelled).
**Boxes only need to be roughly right**: a prediction matches a label when the
boxes overlap by IoU ≥ 0.5, so ±5 points is fine.

## What to label

**`layout_elements`**: one box per block, with a `class`:

| class | Use for |
|---|---|
| `title` | Document title (usually once, on the first page) |
| `section_header` | Article / section / form-section headings |
| `text` | Paragraphs, footnotes, captions |
| `list` | List items (one element per item) |
| `table` | Whole table (also add it under `tables`) |
| `page_header` / `page_footer` | Running header / footer text |
| `page_number` | A page number on its own |
| `other` | Logos, pictures, checkboxes, anything else |

`text` inside an element is optional (it helps you, metrics don't use it).

**`reading_order`**: element ids in the order a person reads the page. Multi-column
pages: finish a column before starting the next. Headers/footers/page numbers can be left out.

**`tables`**: for each table: `bbox`, `num_rows`, `num_cols`, `header_rows` (row indices),
and `cells` covering **every grid position exactly once**. A merged cell has
`row_span`/`col_span` > 1 and covers the positions it spans; empty cells still need
an entry with `"text": ""`. Cell text should match the page exactly (numbers especially).

**`key_values`**: label/value pairs on forms, e.g. `{"key": "Loan Amount", "value": "$162,000"}`.
Docling extracts none on these (flat) PDFs, so on CFPB pages you add them all.
EDGAR pages usually have none.

**`header_footer_lines`**: the exact text of every running header/footer line on the page.

**Scanned pages** (`scan_NN_p001`): label what is visible on the image. Reference text
comes from the source page automatically (`reference_text_source: source_page_text_layer`);
don't type it.

## Validator (F4)

`labeling validate` fails on: overlapping or missing table cells, cells outside the grid,
reading-order ids that don't exist or repeat, boxes outside the page, unknown classes,
and verified pages without `labeled_by`.
