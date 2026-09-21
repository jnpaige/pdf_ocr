"""
locate_and_annotate.py — optional post-process step for anything downstream
of pdf_ocr (site_vocab_extractor, site_coder, site_attribute_extractor, ...)
that wants its extracted mentions shown as colored boxes on the source PDF.

This is deliberately decoupled from the extraction tools themselves: they
keep writing exactly the output they already do (a CSV of
item/unit_label/.../key/value rows, per the run-metadata convention shared
across this tool family), and this script consumes that CSV as a second,
independent pass. Nothing about running it is required — it's an opt-in
visual QA layer on top of an extraction that already happened.

Two stages:

1. Locate — for each extraction row, resolve its unit_label to a set of
   candidate pages (via the segmenter's segments.json: block-level
   "T1N_R2E"-style labels use that segment's `pages`; section-level
   "T1N_R2E_S30"-style labels use that section's own `pages`), then
   fuzzy-match the row's `value` string against sliding windows of that
   page's OCR cells (from pdf_ocr's ocr_docling.json, which now carries a
   real bbox per line). This is an easier match than typical fuzzy matching
   sounds like, because extraction prompts already require the verbatim
   OCR'd string, not a normalized one — so you're matching raw OCR text
   against the exact cells that produced it.

   Writes located_mentions.csv — every extraction row, plus (if matched)
   page_index/bbox/matched_text/score. Unmatched rows are kept with those
   columns blank rather than dropped — same "don't force a guess" principle
   used everywhere else in this pipeline family.

2. Annotate — groups matches by (item, page_index) and draws a colored,
   semi-transparent box at each mention's location directly on that item's
   <item>_ocr.pdf (same coordinate space Surya/Docling already computed
   bboxes in, so no separate page-image rendering is needed). Writes
   <item>_annotated.pdf per document that had any matches, in output_dir —
   never overwrites the original _ocr.pdf.

Config (YAML):
    ocr_output_dir:   pdf_ocr's per-document output folders
    segments_dir:     site_form_segmenter's run folder (segments.json per item)
    extraction_csv:   the downstream tool's output CSV to visualize
    output_dir:       where to write located_mentions.csv and annotated PDFs
    columns:          {item: item, unit_label: unit_label, key: key, value: value}
                       — override if a tool's CSV uses different column names
    layers:           true to write toggleable PDF layers (optional content
                       groups) instead of one flat plane — "01 Segments /
                       <label>" margin bars showing which spatial unit each
                       page was assigned to, and "02 Mentions / <category>"
                       boxes+markers per category. Output is named
                       <item>_layers.pdf rather than <item>_annotated.pdf.
                       Default false (flat mode, unchanged).
    output_pages:     "all" (default) or "LO-HI" / [LO, HI] to keep only that
                       0-indexed page range in the written PDF. Trimming happens
                       after drawing, and the baked-in "=== Page N ===" stamps
                       keep their ORIGINAL numbers, so a trimmed file still lines
                       up with the .md and segments.json.
    match_threshold:  rapidfuzz score cutoff, 0-100 (default 75)
    category_colors:  {key_value: [r, g, b]} 0-1 floats — optional, falls
                       back to a default palette cycling through the keys
                       actually present in the data

Usage:
    uv run python locate_and_annotate.py --config config_locate_annotate.yaml
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml
from rapidfuzz import fuzz

MAX_WINDOW_CELLS = 4
UNIT_LABEL_RE = re.compile(r"^T(\w+)_R(\w+)(?:_S(\d+))?$", re.IGNORECASE)

DEFAULT_PALETTE = [
    (0.90, 0.20, 0.20), (0.20, 0.55, 0.90), (0.20, 0.75, 0.30),
    (0.90, 0.60, 0.10), (0.60, 0.25, 0.80), (0.10, 0.75, 0.75),
    (0.85, 0.30, 0.60), (0.55, 0.55, 0.15),
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _parse_page_range(spec) -> tuple[int, int] | None:
    """Parse an `output_pages` spec like "0-35" into (lo, hi) inclusive.

    None/"all"/empty means keep every page. Trimming is safe for the page-index
    contract even though it renumbers the PDF's own pages: pdf_ocr bakes the
    `=== Page N ===` marker into each page as real text, so a trimmed page still
    carries its ORIGINAL index and still lines up with the .md, text_docling.txt
    and segments.json. Useful when a pilot annotates 36 pages of a 189-page,
    144MB volume and you want a file that opens quickly.
    """
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in {"", "all"}):
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
    else:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(spec))
        if not m:
            raise ValueError(f"output_pages must be 'all', 'LO-HI', or [LO, HI] — got {spec!r}")
        lo, hi = int(m.group(1)), int(m.group(2))
    if lo < 0 or hi < lo:
        raise ValueError(f"output_pages range is invalid: {spec!r}")
    return lo, hi


# ---------------------------------------------------------------------------
# Stage 1: locate
# ---------------------------------------------------------------------------

def _load_segments_pages(segments_path: Path) -> dict[str, list[int]]:
    """Returns {unit_label -> pages}, covering both block-level labels
    ("T1N_R2E") and section-level labels ("T1N_R2E_S30")."""
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for seg in data.get("segments", []):
        label = seg["label"]
        out[label] = seg.get("pages", [])
        for sec in seg.get("sections", []):
            out[f"{label}_S{sec['section']}"] = sec.get("pages", [])
    return out


def _resolve_unit_pages(unit_label: str, segments_pages: dict[str, list[int]]) -> list[int] | None:
    if unit_label in segments_pages:
        return segments_pages[unit_label]
    return None  # unrecognized unit_label shape — leave unresolved, don't guess


def _reading_order_t(bbox: dict) -> float:
    """A "smaller = read first" y-key regardless of coordinate origin.
    TOPLEFT: smaller t is higher on the page (already reading-order).
    BOTTOMLEFT: LARGER t is higher on the page (y increases upward, same
    convention _build_searchable_pdf already relies on: `ph - bbox.t` only
    produces a valid top edge because t is the bigger of the two values) —
    negate so ascending sort still means top-to-bottom."""
    t = bbox["t"]
    return -t if bbox.get("origin") == "BOTTOMLEFT" else t


def _page_cells_sorted(page: dict) -> list[dict]:
    lines = [l for l in page.get("text_lines", []) if l.get("text", "").strip() and l.get("bbox")]
    return sorted(lines, key=lambda l: (round(_reading_order_t(l["bbox"]) / 15.0), l["bbox"]["l"]))


def _union_bbox(cells: list[dict]) -> dict:
    """Union multiple cells' boxes into one spanning box. min/max of t/b
    flips depending on origin — BOTTOMLEFT's "top" (t) is the LARGER
    coordinate (see _reading_order_t) — get this backwards and a multi-cell
    window's bbox comes out inverted/wrong on every Docling-assembled page,
    which is every page that didn't need the raw-OCR fallback."""
    boxes = [c["bbox"] for c in cells]
    origin = boxes[0]["origin"]
    ts = [b["t"] for b in boxes]
    bs = [b["b"] for b in boxes]
    top, bottom = (max(ts), min(bs)) if origin == "BOTTOMLEFT" else (min(ts), max(bs))
    return {
        "l": min(b["l"] for b in boxes), "t": top,
        "r": max(b["r"] for b in boxes), "b": bottom,
        "origin": origin,
    }


def _best_match_on_page(value_norm: str, cells: list[dict], threshold: float) -> dict | None:
    """partial_ratio, not ratio: a "cell" can be a whole Docling-assembled
    paragraph spanning several original OCR lines (merged with embedded
    newlines), so a short target is often a substring of a much longer
    candidate window rather than a near-equal-length string — ratio
    penalizes that length mismatch even for a perfect substring match,
    partial_ratio finds the best-aligning substring instead."""
    best = None
    n = len(cells)
    min_window_len = max(3, int(len(value_norm) * 0.6))
    for i in range(n):
        for length in range(1, min(MAX_WINDOW_CELLS, n - i) + 1):
            window = cells[i:i + length]
            window_text = " ".join(c["text"] for c in window)
            window_norm = _normalize(window_text)
            # partial_ratio degenerates against a much-shorter candidate — it
            # finds the best alignment of the SHORTER string's length, so a
            # 1-character window vs. a 9-character target can score a
            # trivial "perfect" match. Require the window to be a real
            # fraction of the target's length before it's even considered.
            if len(window_norm) < min_window_len:
                continue
            score = fuzz.partial_ratio(value_norm, window_norm)
            if score >= threshold and (best is None or score > best["score"]):
                best = {"score": score, "matched_text": window_text, "bbox": _union_bbox(window)}
    return best


def locate(cfg: dict, repo_root: Path) -> list[dict]:
    cols = cfg.get("columns", {})
    item_col = cols.get("item", "item")
    unit_col = cols.get("unit_label", "unit_label")
    key_col = cols.get("key", "key")
    value_col = cols.get("value", "value")
    page_col = cols.get("page", "page")
    threshold = float(cfg.get("match_threshold", 75))

    ocr_output_dir = Path(cfg["ocr_output_dir"])
    segments_dir = Path(cfg["segments_dir"])

    rows = list(csv.DictReader(open(cfg["extraction_csv"], encoding="utf-8")))
    print(f"{len(rows)} extraction rows to locate")

    rows_by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_item[r[item_col]].append(r)

    located: list[dict] = []
    n_matched = 0
    n_via_reported_page = 0
    for item, item_rows in rows_by_item.items():
        ocr_json_path = ocr_output_dir / item / "ocr_docling.json"
        segments_path = next(segments_dir.glob(f"*{item}*.segments.json"), None)
        if not ocr_json_path.exists() or segments_path is None:
            print(f"  SKIP {item} — missing ocr_docling.json or segments.json")
            for r in item_rows:
                located.append({**r, "page_index": "", "bbox": "", "matched_text": "",
                                "match_score": "", "location_method": ""})
            continue

        pages = json.loads(ocr_json_path.read_text(encoding="utf-8"))
        pages_by_index = {p["page_index"]: p for p in pages}
        segments_pages = _load_segments_pages(segments_path)

        for r in item_rows:
            value_norm = _normalize(r[value_col])

            # Fast path: the extractor's own self-reported page (see
            # extract.py's _normalize_page_tagged — validated there against
            # the pages actually sent to the LLM, so a value here is already
            # trustworthy, not a raw guess). Search only that one page first
            # — this is what disambiguates two genuine, differently-located
            # occurrences of the same word within one unit (e.g. "Hackberry"
            # in an unrelated citation on page 1 vs. a real compound species
            # list on page 11) that whole-unit search can't tell apart.
            best = None
            best_page = None
            method = ""
            reported_page_raw = (r.get(page_col) or "").strip()
            if reported_page_raw.isdigit():
                reported_page = int(reported_page_raw)
                page = pages_by_index.get(reported_page)
                if page:
                    cells = _page_cells_sorted(page)
                    match = _best_match_on_page(value_norm, cells, threshold)
                    if match:
                        best, best_page, method = match, reported_page, "self_reported_page"

            # Fallback: no reported page, or it didn't actually contain a
            # match (never worse than before this change) — search every
            # page the value's unit could plausibly be on, same as always.
            if best is None:
                candidate_pages = _resolve_unit_pages(r[unit_col], segments_pages) or []
                for pg in candidate_pages:
                    page = pages_by_index.get(pg)
                    if not page:
                        continue
                    cells = _page_cells_sorted(page)
                    match = _best_match_on_page(value_norm, cells, threshold)
                    if match and (best is None or match["score"] > best["score"]):
                        best, best_page, method = match, pg, "unit_wide_search"

            if best:
                n_matched += 1
                if method == "self_reported_page":
                    n_via_reported_page += 1
                located.append({
                    **r, "page_index": best_page, "bbox": json.dumps(best["bbox"]),
                    "matched_text": best["matched_text"], "match_score": best["score"],
                    "location_method": method,
                })
            else:
                located.append({**r, "page_index": "", "bbox": "", "matched_text": "",
                                "match_score": "", "location_method": ""})

    print(f"Located {n_matched}/{len(rows)} rows ({100 * n_matched / max(len(rows), 1):.1f}%)"
          f" — {n_via_reported_page} via self-reported page, {n_matched - n_via_reported_page} via unit-wide search")

    out_path = Path(cfg["output_dir"]) / "located_mentions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) + ["page_index", "bbox", "matched_text", "match_score", "location_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(located)
    print(f"-> {out_path}")
    return located


# ---------------------------------------------------------------------------
# Stage 2: annotate
# ---------------------------------------------------------------------------

def _bbox_area(b: dict) -> float:
    return abs(b["r"] - b["l"]) * abs(b["t"] - b["b"])


def _bboxes_overlap(a: dict, b: dict, threshold: float = 0.15) -> bool:
    """True if the two boxes overlap by at least `threshold` fraction of the
    smaller box's area. Docling merges a whole paragraph into one item, so
    it's common for several different mentions in the same paragraph to all
    resolve to the same (or a heavily overlapping) window — this is what
    lets those share one drawn box instead of stacking rectangles that hide
    each other. Kept deliberately low (not e.g. 0.5): two mentions from the
    same paragraph don't always land on identical windows — one might match
    a 1-cell span and another a 2-cell span that only partially covers it —
    and a strict threshold left those as separate, visually-overlapping
    boxes that still competed with each other on screen."""
    l, r = max(a["l"], b["l"]), min(a["r"], b["r"])
    top = max(min(a["t"], a["b"]), min(b["t"], b["b"]))
    bottom = min(max(a["t"], a["b"]), max(b["t"], b["b"]))
    if r <= l or bottom <= top:
        return False
    inter = (r - l) * (bottom - top)
    return inter / min(_bbox_area(a), _bbox_area(b)) >= threshold


def _cluster_mentions(mentions: list[dict]) -> list[list[dict]]:
    """Union-find over pairwise bbox overlap, within one page's mentions."""
    n = len(mentions)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boxes = [json.loads(m["bbox"]) for m in mentions]
    for i in range(n):
        for j in range(i + 1, n):
            if _bboxes_overlap(boxes[i], boxes[j]):
                union(i, j)

    groups: dict[int, list[dict]] = defaultdict(list)
    for i, m in enumerate(mentions):
        groups[find(i)].append(m)
    return list(groups.values())


def _union_raw_bboxes(boxes: list[dict]) -> dict:
    origin = boxes[0]["origin"]
    ts = [b["t"] for b in boxes]
    bs = [b["b"] for b in boxes]
    top, bottom = (max(ts), min(bs)) if origin == "BOTTOMLEFT" else (min(ts), max(bs))
    return {
        "l": min(b["l"] for b in boxes), "t": top,
        "r": max(b["r"] for b in boxes), "b": bottom,
        "origin": origin,
    }


def annotate(cfg: dict, located: list[dict]) -> None:
    import fitz

    cols = cfg.get("columns", {})
    item_col = cols.get("item", "item")
    key_col = cols.get("key", "key")
    value_col = cols.get("value", "value")

    ocr_output_dir = Path(cfg["ocr_output_dir"])
    out_dir = Path(cfg["output_dir"])

    palette_cfg = cfg.get("category_colors", {})
    keys_present = sorted({r[key_col] for r in located if r.get("bbox")})
    palette = {
        k: tuple(palette_cfg[k]) if k in palette_cfg else DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
        for i, k in enumerate(keys_present)
    }

    by_item_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in located:
        if not r.get("bbox"):
            continue
        by_item_page[(r[item_col], int(r["page_index"]))].append(r)

    # Marker grid geometry for mentions that share one paragraph box —
    # small colored squares stepping across then down from the box's
    # top-left, one per mention, each individually hoverable.
    MARKER = 9
    GAP = 2
    MAX_COLS = 5

    def _to_fitz_rect(bbox: dict, page_height: float) -> "fitz.Rect":
        if bbox.get("origin") == "BOTTOMLEFT":
            x0, y0 = bbox["l"], page_height - bbox["t"]
            x1, y1 = bbox["r"], page_height - bbox["b"]
        else:
            x0, y0, x1, y1 = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
        return fitz.Rect(x0, y0, x1, y1)

    # --- Optional OCG layer mode -------------------------------------------
    # Off by default, so the existing flat output is unchanged. When on, every
    # category owns a toggleable PDF layer, which is what lets colour mean
    # something again: the flat plane below had to give up per-category box
    # colour entirely, because a paragraph mentioning both a village and a road
    # could only get one box and differently-coloured boxes overdrew each other
    # (see the neutral-outline comment further down). Layers remove that
    # constraint — a paragraph in three categories simply appears in three
    # planes. Same reasoning annotate_layers.py gives for the lithic pipeline.
    use_layers = bool(cfg.get("layers", False))
    page_range = _parse_page_range(cfg.get("output_pages"))
    segments_dir = Path(cfg["segments_dir"]) if cfg.get("segments_dir") else None

    n_annotated_docs = 0
    for item in sorted({item for item, _ in by_item_page}):
        ocr_pdf_path = ocr_output_dir / item / f"{item}_ocr.pdf"
        if not ocr_pdf_path.exists():
            print(f"  SKIP {item} — no {ocr_pdf_path.name}")
            continue

        doc = fitz.open(str(ocr_pdf_path))
        n_src_pages = len(doc)

        # OCGs are per-document, so build them per item.
        _ocg_cache: dict[str, int] = {}

        def ocg(name: str, on: bool = True):
            """xref of the named layer, or None when layer mode is off."""
            if not use_layers:
                return None
            if name not in _ocg_cache:
                _ocg_cache[name] = doc.add_ocg(name, on=on)
            return _ocg_cache[name]

        # Layer 01 — the SEGMENTATION itself: a margin bar on every page,
        # one layer per township/range block. This is what makes the whole
        # chain legible in one file: which spatial unit a page was assigned
        # to, and which mentions were found inside it.
        if use_layers and segments_dir:
            seg_path = next(segments_dir.glob(f"*{item}*.segments.json"), None)
            if seg_path:
                seg_pages = _load_segments_pages(seg_path)
                for label in sorted(seg_pages):
                    if label.endswith("_pages") or not seg_pages[label]:
                        continue
                    xref = ocg(f"01 Segments / {label}", on=True)
                    for pno in seg_pages[label]:
                        if pno >= len(doc):
                            continue
                        pg = doc[pno]
                        bar = fitz.Rect(6, 60, 18, pg.rect.height - 60)
                        try:
                            pg.draw_rect(bar, color=(0.15, 0.35, 0.65),
                                         fill=(0.15, 0.35, 0.65), fill_opacity=0.30,
                                         width=0, overlay=True, oc=xref)
                        except Exception:
                            pass

        # Stamp every page (not just pages with mentions) with the same
        # `=== Page N ===` marker the .md/.txt outputs use, top-right, as
        # real selectable text — not an annotation — so a page can be
        # matched back to its .md/text_docling.txt section by eye or by
        # copy-pasting straight out of a PDF viewer. pdf_ocr itself now
        # bakes this into every fresh _ocr.pdf (_build_searchable_pdf), so
        # skip pages that already carry it rather than stamping twice —
        # this still covers older _ocr.pdf files from before that existed.
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            label = f"=== Page {page_idx} ==="
            if label in page.get_text():
                continue
            fontsize = 9
            text_width = fitz.get_text_length(label, fontsize=fontsize)
            try:
                page.insert_text(
                    fitz.Point(page.rect.width - text_width - 15, 25),
                    label, fontsize=fontsize, color=(0.2, 0.2, 0.2), overlay=True,
                )
            except Exception:
                continue

        n_clusters = 0
        n_markers = 0
        for (it, page_idx), mentions in by_item_page.items():
            if it != item or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            ph = page.rect.height

            for cluster in _cluster_mentions(mentions):
                boxes = [json.loads(m["bbox"]) for m in cluster]
                rect = _to_fitz_rect(_union_raw_bboxes(boxes), ph)
                cats_here = sorted({m[key_col] for m in cluster})
                if use_layers:
                    # One box per category present, each in its own plane.
                    # Nested slightly so co-located categories stay visible
                    # when several layers are switched on at once.
                    for depth, cat in enumerate(cats_here):
                        xref = ocg(f"02 Mentions / {cat}", on=True)
                        r2 = fitz.Rect(rect.x0 - depth * 1.6, rect.y0 - depth * 1.6,
                                       rect.x1 + depth * 1.6, rect.y1 + depth * 1.6)
                        try:
                            page.draw_rect(r2, color=palette.get(cat, (0.5, 0.5, 0.5)),
                                           width=2.0, stroke_opacity=0.85,
                                           overlay=True, oc=xref)
                            n_clusters += 1
                        except Exception:
                            continue
                else:
                    try:
                        # Flat mode: border only, no fill, and ALWAYS the same
                        # neutral color — not colored by the cluster's
                        # "dominant" category. A paragraph mentioning both a
                        # village and a road used to get one box in whichever
                        # category happened to have more mentions, silently
                        # hiding that the other category was there at all (and
                        # when two categories' matches didn't quite cluster
                        # together, their differently-colored boxes would
                        # visually compete, each overdrawing the other). One
                        # consistent outline color can never hide another box,
                        # since there's only ever one color — the marker grid
                        # below is the only place category identity lives in
                        # this mode, and it's already laid out so nothing
                        # overlaps. Layer mode has no such constraint, which is
                        # why it colors the boxes above.
                        page.draw_rect(rect, color=(0.35, 0.35, 0.35), width=3.6,
                                       stroke_opacity=0.5, overlay=True)
                        n_clusters += 1
                    except Exception:
                        continue

                # Group same-category mentions together in the grid so the
                # per-category count is easy to read off at a glance
                # (e.g. "3 green squares then 2 blue" rather than colors
                # interleaved in arbitrary extraction order).
                cluster_sorted = sorted(cluster, key=lambda m: m[key_col])
                for idx, m in enumerate(cluster_sorted):
                    row, col = divmod(idx, MAX_COLS)
                    mx0 = rect.x0 + col * (MARKER + GAP)
                    my0 = rect.y0 + row * (MARKER + GAP)
                    marker_rect = fitz.Rect(mx0, my0, mx0 + MARKER, my0 + MARKER)
                    m_color = palette.get(m[key_col], (0.5, 0.5, 0.5))
                    m_xref = ocg(f"02 Mentions / {m[key_col]}", on=True)
                    try:
                        page.draw_rect(marker_rect, color=m_color, fill=m_color,
                                        fill_opacity=0.9, width=0.5, overlay=True,
                                        oc=m_xref)
                        annot = page.add_text_annot(marker_rect.tl, f"{m[key_col]}: {m[value_col]}")
                        annot.set_colors(stroke=m_color)
                        if m_xref is not None:
                            annot.set_oc(m_xref)
                        annot.update()
                        n_markers += 1
                    except Exception:
                        continue

        out_path = out_dir / (f"{item}_layers.pdf" if use_layers
                              else f"{item}_annotated.pdf")
        # Trim LAST: every page index used above is an index into the untrimmed
        # document, so selecting earlier would shift pages under the annotations.
        if page_range:
            lo, hi = page_range
            keep = [i for i in range(len(doc)) if lo <= i <= hi]
            if not keep:
                print(f"  SKIP {item} — output_pages {lo}-{hi} selects no pages")
                doc.close()
                continue
            doc.select(keep)
            print(f"     trimmed to pages {lo}-{hi} ({len(keep)} of {n_src_pages}); "
                  f"'=== Page N ===' stamps still carry original indices")
        doc.save(str(out_path), garbage=3, deflate=True)
        doc.close()
        n_annotated_docs += 1
        print(f"  OK {item} — {n_clusters} paragraph box(es), {n_markers} mention marker(s) -> {out_path.name}")

    print(f"\nAnnotated {n_annotated_docs} document(s)")
    if keys_present:
        print("Legend:", ", ".join(f"{k}={palette[k]}" for k in keys_present))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_locate_annotate.yaml", metavar="PATH")
    ap.add_argument("--locate-only", action="store_true", help="Only write located_mentions.csv, skip annotation")
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    located = locate(cfg, repo_root)
    if not args.locate_only:
        annotate(cfg, located)


if __name__ == "__main__":
    main()
