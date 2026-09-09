#!/usr/bin/env python3
"""
annotate_layers.py — draw the pipeline's own output back onto the OCR'd PDF as
toggleable PDF layers (optional content groups), one per kind of information.

Companion to locate_and_annotate.py, and deliberately a separate script rather
than an extension of it: that one visualizes an extraction CSV of short verbatim
mentions located by fuzzy matching, this one visualizes whole-document structure
(pass0 sections, parsed blocks, cluster tags, classified figures) whose geometry
is already known exactly. Same principles though — opt-in, reads what the other
tools already wrote, and never overwrites <stem>_ocr.pdf.

WHY LAYERS. locate_and_annotate learned the hard way that multi-label data drawn
into a single always-visible plane defeats itself: a paragraph belonging to two
categories got one box in whichever category had more mentions, silently hiding
that the other was there, and boxes of different colors overdrew each other. Its
fix was to give up per-category color for the box outline entirely and move
category identity into a grid of little marker squares. Optional content groups
remove the constraint that forced that compromise — each category owns a plane
the reader can switch on and off, so overlapping boxes never compete and color
means something again. A block tagged to three clusters simply appears in three
layers.

LAYERS PRODUCED (ordered by name so viewers group them sensibly):
    01 Structure / <pass0 section>   margin bar per page, per section category
    01 Structure / dropped by pass0  pages pass0 excluded from tagging
    02 Blocks / headings|paragraphs|table rows   what the block parser produced
    03 Clusters / <assemblage label>  blocks tagged to that cluster   [on]
    03 Clusters / comparative (2+) | general | untagged
    04 Figures / <classification>     figure regions by class

Only the cluster layers are on by default; everything else is a diagnostic the
reader switches on when they want it.

INPUTS (all already written by the existing pipeline)
    <doc>/<stem>.cluster_tags.json   pilot_cluster_tagging.py — blocks + tags + pass0
    <doc>/ocr_docling.json           pdf_ocr — per-item bboxes, and per-table
                                     row/cell geometry where the corpus was OCR'd
                                     after table export was added
    <doc>/figures/figures.json       pdf_ocr — figure bbox + classification
    <doc>/<stem>_ocr.pdf             the page images to draw on

Usage:
    uv run python annotate_layers.py --doc-dir "path/to/<document folder>"
    uv run python annotate_layers.py --input-dir "path/to/corpus"   # every doc in it
"""
import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CLUSTER_PALETTE = [
    (0.85, 0.15, 0.15), (0.15, 0.45, 0.85), (0.15, 0.65, 0.30),
    (0.90, 0.55, 0.05), (0.55, 0.20, 0.75), (0.05, 0.65, 0.68),
    (0.85, 0.25, 0.55), (0.50, 0.50, 0.10),
]
SECTION_COLOR = (0.35, 0.45, 0.60)
DROPPED_COLOR = (0.80, 0.20, 0.20)
BLOCK_TYPE_COLORS = {
    "heading": (0.20, 0.30, 0.70),
    "paragraph": (0.45, 0.45, 0.45),
    "table_row": (0.20, 0.55, 0.35),
}
FIGURE_COLOR = (0.60, 0.35, 0.75)

# An OCR item shorter than this is too generic to trust as a containment match
# for a block ("Table 2.", a stray letter) — matched by equality only.
MIN_CONTAINMENT_CHARS = 20
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Normalize for matching block text against OCR item text: both sides may
    carry markdown heading markers and entity-escaped inline markup."""
    s = _TAG_RE.sub("", html.unescape(s or ""))
    s = re.sub(r"^#{1,6}\s*", "", s.strip())
    return _WS_RE.sub(" ", s).strip().lower()


def _row_cells(block_text: str) -> list[str]:
    """Cell strings from a table_row block ('columns: ... || row: | a | b |')."""
    part = block_text.split("||", 1)[-1]
    part = part.split(":", 1)[-1]
    return [c.strip().lower() for c in part.split("|") if c.strip()]


# ---------------------------------------------------------------------------
# Geometry resolution — block -> rectangles on the page
# ---------------------------------------------------------------------------

def build_index(ocr_json: list[dict]) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    items: dict[int, list[dict]] = defaultdict(list)
    tables: dict[int, list[dict]] = defaultdict(list)
    for page in ocr_json:
        pi = page.get("page_index", 0)
        for line in page.get("text_lines", []):
            if not (line.get("text") or "").strip():
                continue
            # `bboxes` carries every region the item occupies, each with its own
            # page — a paragraph flowing across a column or page break has more
            # than one. Fall back to the single `bbox` for output written before
            # that existed, so older corpora still resolve (to their first
            # region only, which is what was recorded).
            regions = [(r.get("page", pi), r) for r in (line.get("bboxes") or [])]
            if not regions and line.get("bbox"):
                regions = [(pi, line["bbox"])]
            if regions:
                items[pi].append({"norm": _norm(line["text"]), "regions": regions})
        for table in page.get("tables", []):
            tables[pi].append(table)
    return items, tables


def resolve_text_block(block: dict, items: dict[int, list[dict]]) -> list[tuple[int, dict]]:
    """(page_index, rectangle) pairs for a heading/paragraph block.

    Returns a list, not one box, for two independent reasons that both produce
    multi-region blocks in a two-column paper. stitch_continuations merges a
    paragraph interrupted by a floated table back into one block, so the block
    covers two separate regions; and docling itself records one `prov` region
    per column for a paragraph that flows across a column or page break. The
    page travels with each rectangle because in both cases a later region can
    sit on a page the block itself doesn't name.
    """
    target = _norm(block["text"])
    if not target:
        return []
    page = block["page"]
    found: list[tuple[int, dict]] = []
    for pi in (page, page + 1):
        for item in items.get(pi, []):
            n = item["norm"]
            if not n:
                continue
            if n == target or (len(n) >= MIN_CONTAINMENT_CHARS and n in target):
                found.extend(item["regions"])
    return found


def resolve_table_row(block: dict, tables: dict[int, list[dict]]) -> list[tuple[int, dict]]:
    """Rectangle for a table_row block, matched to a docling table row by cell
    text rather than by row index — the markdown a block was parsed from treats
    only the first pipe line as a header, while docling may mark several rows as
    column headers, so positional alignment is not guaranteed to hold."""
    cells = _row_cells(block["text"])
    if not cells:
        return []
    best, best_score = None, 0.0
    for table in tables.get(block["page"], []):
        by_row: dict[int, list[str]] = defaultdict(list)
        for cell in table.get("cells", []):
            by_row[cell["start_row"]].append((cell.get("text") or "").strip().lower())
        for row in table.get("rows", []):
            if not row.get("bbox"):
                continue
            row_cells = [c for c in by_row.get(row["row"], []) if c]
            if not row_cells:
                continue
            hits = sum(1 for c in cells if c in row_cells)
            score = hits / max(len(cells), 1)
            if score > best_score:
                best, best_score = row["bbox"], score
    return [(block["page"], best)] if best is not None and best_score >= 0.5 else []


def to_rect(bbox: dict, page_height: float):
    import fitz
    if bbox.get("origin") == "BOTTOMLEFT":
        return fitz.Rect(bbox["l"], page_height - bbox["t"], bbox["r"], page_height - bbox["b"])
    return fitz.Rect(bbox["l"], bbox["t"], bbox["r"], bbox["b"])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def annotate_document(doc_dir: Path, out_dir: Path | None, verbose: bool = True) -> Path | None:
    import fitz

    stem = doc_dir.name
    tags_path = next(doc_dir.glob("*.cluster_tags.json"), None)
    ocr_path = doc_dir / "ocr_docling.json"
    pdf_path = doc_dir / f"{stem}_ocr.pdf"
    figs_path = doc_dir / "figures" / "figures.json"

    if tags_path is None or not ocr_path.exists() or not pdf_path.exists():
        missing = [n for n, p in [("cluster_tags", tags_path), ("ocr_docling.json", ocr_path if ocr_path.exists() else None),
                                  (pdf_path.name, pdf_path if pdf_path.exists() else None)] if p is None]
        print(f"  SKIP {stem} — missing {', '.join(missing)}")
        return None

    tags_data = json.loads(tags_path.read_text(encoding="utf-8"))
    ocr_json = json.loads(ocr_path.read_text(encoding="utf-8"))
    items, tables = build_index(ocr_json)

    blocks = tags_data.get("blocks", [])
    by_id = {b["block_id"]: b for b in blocks}
    tag_by_id = {t["block_id"]: t for t in tags_data.get("block_tags", []) if t.get("block_id")}
    clusters = tags_data.get("clusters", [])
    pass0 = tags_data.get("pass0_result") or {}

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)

    # --- declare layers -----------------------------------------------------
    ocg: dict[str, int] = {}

    def layer(name: str, on: bool = False) -> int:
        if name not in ocg:
            ocg[name] = doc.add_ocg(name, on=on)
        return ocg[name]

    cluster_color: dict[str, tuple] = {}
    cluster_layer: dict[str, str] = {}
    for i, c in enumerate(clusters):
        cid = c.get("cluster_id") or f"c{i+1}"
        label = (c.get("label_as_reported") or cid).strip() or cid
        cluster_color[cid] = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        cluster_layer[cid] = f"03 Clusters / {label}"
        layer(cluster_layer[cid], on=True)

    counts: dict[str, int] = defaultdict(int)
    unresolved: dict[str, int] = defaultdict(int)

    def draw(page, bbox, color, name, width=1.6, dashes=None):
        rect = to_rect(bbox, page.rect.height)
        if rect.is_empty or rect.is_infinite:
            return False
        try:
            page.draw_rect(rect, color=color, width=width, dashes=dashes,
                           stroke_opacity=0.85, oc=layer(name), overlay=True)
        except Exception:
            return False
        counts[name] += 1
        return True

    # --- 01 Structure: pass0 section map, as margin bars ---------------------
    kept_pages = {b["page"] for b in blocks}
    section_pages: dict[int, list[str]] = defaultdict(list)
    for category, entries in pass0.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                for p in entry.get("pages", []):
                    if category not in section_pages[int(p)]:
                        section_pages[int(p)].append(category)

    for pi in range(n_pages):
        page = doc[pi]
        for slot, category in enumerate(section_pages.get(pi, [])):
            name = f"01 Structure / {category}"
            x = 4 + slot * 7
            try:
                page.draw_rect(fitz.Rect(x, 40, x + 5, page.rect.height - 40),
                               color=SECTION_COLOR, fill=SECTION_COLOR, fill_opacity=0.45,
                               width=0, oc=layer(name), overlay=True)
                counts[name] += 1
            except Exception:
                pass
        if pi not in kept_pages:
            name = "01 Structure / dropped by pass0"
            try:
                page.draw_rect(fitz.Rect(2, 30, page.rect.width - 2, page.rect.height - 30),
                               color=DROPPED_COLOR, width=2.5, dashes="[4 3] 0",
                               stroke_opacity=0.8, oc=layer(name), overlay=True)
                page.insert_text(fitz.Point(10, 26), "dropped by pass0", fontsize=8,
                                 color=DROPPED_COLOR, oc=layer(name), overlay=True)
                counts[name] += 1
            except Exception:
                pass

    # --- 02 Blocks + 03 Clusters -------------------------------------------
    for block in blocks:
        pi = block["page"]
        if pi >= n_pages:
            continue
        if block["type"] == "table_row":
            rects = resolve_table_row(block, tables)
        else:
            rects = resolve_text_block(block, items)
        # A later region can name a page past the end of a chunked/truncated PDF.
        rects = [(p, bb) for p, bb in rects if 0 <= p < n_pages]
        if not rects:
            # Separate the misses that matter from the ones that never had
            # geometry to find. Docling emits a placeholder block where it
            # lifted a picture out of the text flow, and a short caption-like
            # label beside it; both are represented by the figure layers
            # instead. Counting them as failures would bury a real miss in
            # noise once this runs over a corpus.
            text = (block["text"] or "").strip()
            if text.startswith("<!--"):
                unresolved["placeholder (expected)"] += 1
            elif len(text.split()) < 6:
                unresolved["short unmatched (likely image label)"] += 1
            else:
                unresolved[block["type"]] += 1
            continue

        type_name = {"heading": "02 Blocks / headings",
                     "paragraph": "02 Blocks / paragraphs",
                     "table_row": "02 Blocks / table rows"}[block["type"]]
        for rect_page, bbox in rects:
            draw(doc[rect_page], bbox, BLOCK_TYPE_COLORS[block["type"]], type_name, width=0.8)

        tag = tag_by_id.get(block["block_id"], {})
        cids = [c for c in (tag.get("cluster_ids") or []) if c in cluster_layer]
        for cid in cids:
            name = cluster_layer[cid]
            color = cluster_color[cid]
            for rect_page, bbox in rects:
                page = doc[rect_page]
                if draw(page, bbox, color, name, width=1.8):
                    try:
                        annot = page.add_text_annot(
                            to_rect(bbox, page.rect.height).tl,
                            f"{block['block_id']} [{block['type']}]\nclusters: {', '.join(cids)}",
                        )
                        annot.set_colors(stroke=color)
                        annot.set_oc(ocg[name])
                        annot.update()
                    except Exception:
                        pass
        for rect_page, bbox in rects:
            page = doc[rect_page]
            if len(cids) > 1:
                draw(page, bbox, (0.95, 0.45, 0.0), "03 Clusters / comparative (2+)", width=2.6)
            if tag.get("general"):
                draw(page, bbox, (0.25, 0.25, 0.25), "03 Clusters / general", width=1.6)
            if not cids and not tag.get("general"):
                draw(page, bbox, (0.6, 0.6, 0.6), "03 Clusters / untagged", width=1.0, dashes="[2 2] 0")

    # --- 04 Figures ---------------------------------------------------------
    if figs_path.exists():
        try:
            figs = json.loads(figs_path.read_text(encoding="utf-8"))
            figs = figs if isinstance(figs, list) else figs.get("figures", [])
        except Exception:
            figs = []
        for fig in figs:
            pi, bb = fig.get("page"), fig.get("bbox")
            if pi is None or not bb or pi >= n_pages:
                continue
            bbox = ({"l": bb[0], "t": bb[1], "r": bb[2], "b": bb[3], "origin": "BOTTOMLEFT"}
                    if isinstance(bb, list) else bb)
            cls = fig.get("classification") or "unclassified"
            draw(doc[pi], bbox, FIGURE_COLOR, f"04 Figures / {cls}", width=2.0)

    out_dir = out_dir or doc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_layers.pdf"
    doc.save(str(out_path))
    doc.close()

    if verbose:
        print(f"  {stem}")
        print(f"    layers: {len([k for k in ocg if not k.startswith('__')])}   "
              f"boxes: {sum(counts.values())}")
        for name in sorted(k for k in counts):
            print(f"      {name:44s} {counts[name]}")
        if unresolved:
            total = sum(unresolved.values())
            print(f"    no geometry for {total} block(s): " +
                  ", ".join(f"{k}={v}" for k, v in sorted(unresolved.items())))
            if unresolved.get("table_row") and not any(tables.values()):
                print("      table rows have no geometry because this document's "
                      "ocr_docling.json predates table export — re-run pdf_ocr to gain it")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-dir", help="One pdf_ocr output folder")
    g.add_argument("--input-dir", help="Corpus folder; annotates every document in it")
    ap.add_argument("--out-dir", default=None, help="Where to write (default: alongside the source)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.doc_dir:
        dirs = [Path(args.doc_dir)]
    else:
        root = Path(args.input_dir)
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and not d.name[0].isdigit() and any(d.glob("*.cluster_tags.json")))
    if not dirs:
        sys.exit("No documents with a *.cluster_tags.json found")

    written = 0
    for d in dirs:
        if annotate_document(d, out_dir):
            written += 1
    print(f"\nWrote {written} annotated PDF(s)")


if __name__ == "__main__":
    main()
