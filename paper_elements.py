#!/usr/bin/env python3
"""Write <doc>/paper_elements.json: every addressable element of a paper, in one standard shape.

Usage:
    uv run python paper_elements.py --input-dir "<corpus>"
    uv run python paper_elements.py --input-dir "<corpus>" --only "Tryon"
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from block_parser import is_caption, is_placeholder, split_blocks, stitch_continuations
from annotate_layers import build_index, resolve_table_row, resolve_text_block

SCHEMA = "paper_elements/1"
PAGE_MARK = re.compile(r"^=== Page (\d+) ===$", re.MULTILINE)
CAPTION_NUMBER = re.compile(r"^(?:extended\s+data\s+)?fig(?:ure)?s?\.?\s*(\d+)", re.I)
FIGURE_REF = re.compile(r"\bfig(?:ure)?s?\.?\s*((?:\d+[a-z]?(?:\s*(?:[-–—]|,|and|&)\s*)?)+)", re.I)


def parse_pages(md_path: Path) -> dict[int, str]:
    """Split the .md on its === Page N === markers."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    marks = list(PAGE_MARK.finditer(text))
    return {int(m.group(1)): text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip()
            for i, m in enumerate(marks)}


def figure_refs(text: str) -> list[int]:
    """Figure numbers a text cites, with ranges like 'Figs. 2-4' expanded."""
    found = set()
    for m in FIGURE_REF.finditer(text):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        found.update(nums)
        dash = re.search(r"(\d+)[a-z]?\s*[-–—]\s*(\d+)", m.group(1))
        if dash and int(dash.group(1)) < int(dash.group(2)) <= int(dash.group(1)) + 20:
            found.update(range(int(dash.group(1)), int(dash.group(2)) + 1))
    return sorted(found)


def region(page: int, box: dict) -> dict:
    """One rectangle in PDF points, origin recorded."""
    return {"page": page, "l": box["l"], "t": box["t"], "r": box["r"], "b": box["b"],
            "origin": box.get("origin", "BOTTOMLEFT")}


def text_elements(doc_dir: Path) -> list[dict]:
    """Headings, paragraphs, captions and table rows from the .md, with geometry where it resolves."""
    md = next(doc_dir.glob("*.md"))
    blocks, _ = stitch_continuations(split_blocks(parse_pages(md)))
    ocr_path = doc_dir / "ocr_docling.json"
    items, tables = build_index(json.loads(ocr_path.read_text(encoding="utf-8"))) if ocr_path.exists() else ({}, {})
    out, section = [], None
    for b in blocks:
        if is_placeholder(b.text):
            continue
        kind = "caption" if b.type == "paragraph" and is_caption(b.text) else b.type
        if kind == "heading":
            section = b.text
        block = {"page": b.page, "text": b.text}
        boxes = resolve_table_row(block, tables) if kind == "table_row" else resolve_text_block(block, items)
        el = {"id": b.block_id, "kind": kind, "page": b.page, "text": b.text, "section": section,
              "regions": [region(p, box) for p, box in boxes], "figure_refs": figure_refs(b.text)}
        if kind == "caption":
            m = CAPTION_NUMBER.match(b.text.strip())
            el["figure_number"] = int(m.group(1)) if m else None
        out.append(el)
    return out


def figure_elements(doc_dir: Path, captions: dict[int, str]) -> list[dict]:
    """Figures from figures.json and their artefact crops from figure_pieces.json."""
    fig_path = doc_dir / "figures" / "figures.json"
    if not fig_path.exists():
        return []
    figs = json.loads(fig_path.read_text(encoding="utf-8")).get("figures", [])
    pieces_path = doc_dir / "figures" / "figure_pieces.json"
    pieces = {e["file"]: e for e in json.loads(pieces_path.read_text(encoding="utf-8"))} if pieces_path.exists() else {}
    out = []
    for f in figs:
        fid, number = Path(f["file"]).stem, f.get("printed_figure_number")
        bb = f.get("bbox")
        box = {"l": bb[0], "t": bb[1], "r": bb[2], "b": bb[3], "origin": "BOTTOMLEFT"} if isinstance(bb, list) else bb
        out.append({"id": fid, "kind": "figure", "page": f.get("page"), "file": f"figures/{f['file']}",
                    "figure_number": number, "caption": f.get("caption") or captions.get(number),
                    "section": " > ".join(f.get("heading_path") or []) or None,
                    "regions": [region(f.get("page"), box)] if box else []})
        entry = pieces.get(f["file"])
        if entry and box and (entry.get("plate_px") or [0])[0]:
            scale = (box["r"] - box["l"]) / entry["plate_px"][0]
            for i, g in enumerate(entry.get("pieces", []), 1):
                piece_box = {"l": box["l"] + g["l"] * scale, "r": box["l"] + g["r"] * scale,
                             "t": box["t"] - g["t"] * scale, "b": box["t"] - g["b"] * scale}
                out.append({"id": f"{fid}_piece{i:02d}", "kind": "piece", "parent": fid, "page": f.get("page"),
                            "file": f"figures/piece_crops/{entry['crops'][i - 1]}" if entry.get("crops") else None,
                            "plate_box": {k: g[k] for k in ("l", "t", "r", "b")} | {"origin": "TOPLEFT"},
                            "regions": [region(f.get("page"), piece_box)]})
    return out


def build(doc_dir: Path) -> dict:
    texts = text_elements(doc_dir)
    captions = {e["figure_number"]: e["text"] for e in texts if e["kind"] == "caption" and e.get("figure_number")}
    return {"schema": SCHEMA, "paper": doc_dir.name, "n_pages": len(parse_pages(next(doc_dir.glob("*.md")))),
            "elements": texts + figure_elements(doc_dir, captions)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--only", help="substring of one document folder name")
    args = ap.parse_args()
    docs = sorted(d for d in args.input_dir.iterdir() if d.is_dir() and any(d.glob("*.md")))
    for d in docs:
        if args.only and args.only not in d.name:
            continue
        data = build(d)
        (d / "paper_elements.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        kinds = {}
        for e in data["elements"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        located = sum(1 for e in data["elements"] if e["regions"])
        print(f"{d.name[:50]:50} {kinds}  located {located}/{len(data['elements'])}")


if __name__ == "__main__":
    main()
