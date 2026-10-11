#!/usr/bin/env python3
"""Draw a paper_units.json onto the searchable PDF: boxes coloured by assemblage, every figure flagged.

Writes <out-dir>/<paper>_units.pdf, with one toggleable layer per assemblage,
a figure-flag layer, and hidden layers for layer links and unlinked elements.

Usage:
    uv run python annotate_paper_units.py --input-dir "<corpus>" --run "<paper_units run dir>" [--only "<paper>"]
"""
import argparse
import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent))
from annotate_layers import to_rect

PALETTE = [(0.85, 0.15, 0.15), (0.10, 0.45, 0.85), (0.10, 0.65, 0.25), (0.90, 0.55, 0.00),
           (0.55, 0.20, 0.75), (0.00, 0.65, 0.70), (0.85, 0.25, 0.60), (0.45, 0.35, 0.15),
           (0.40, 0.55, 0.00), (0.20, 0.20, 0.55)]
FLAG_COLOR = (0.05, 0.05, 0.05)
UNLINKED_COLOR = (0.6, 0.6, 0.6)
INSET = 1.6


def colour(i: int) -> tuple:
    return PALETTE[i % len(PALETTE)]


def rects(element: dict, page_height: dict) -> list[tuple[int, fitz.Rect]]:
    """(page, rectangle) for each region of an element, in PyMuPDF coordinates."""
    out = []
    for r in element.get("regions") or []:
        if r.get("page") in page_height:
            out.append((r["page"], to_rect(r, page_height[r["page"]])))
    return out


def annotate(paper_dir: Path, units: dict, out_dir: Path) -> Path:
    elements = json.loads((paper_dir / "paper_elements.json").read_text(encoding="utf-8"))["elements"]
    doc = fitz.open(str(next(paper_dir.glob("*_ocr.pdf"))))
    heights = {i: doc[i].rect.height for i in range(len(doc))}
    links = {k["element"]: k for k in units["links"]}
    asm = {a["id"]: (i, a) for i, a in enumerate(units["assemblages"])}
    layers = {a["id"]: doc.add_ocg(f"Assemblage {a['id']}: {a['label']}"[:80], on=True) for a in units["assemblages"]}
    flags = doc.add_ocg("Figure flags", on=True)
    layer_links = doc.add_ocg("Layer links", on=False)
    unlinked = doc.add_ocg("Unlinked elements", on=False)

    for e in elements:
        k = links.get(e["id"], {})
        ids = [a for a in k.get("assemblages", []) if a in asm]
        for page, rect in rects(e, heights):
            for depth, a in enumerate(ids):
                doc[page].draw_rect(rect + (-depth * INSET, -depth * INSET, depth * INSET, depth * INSET),
                                    color=colour(asm[a][0]), width=1.4, oc=layers[a], overlay=True)
            if k.get("layers"):
                doc[page].draw_rect(rect + (-6, 0, -5, 0), color=(0.3, 0.3, 0.3), width=2.5, oc=layer_links, overlay=True)
            if not ids and e["kind"] != "piece":
                doc[page].draw_rect(rect, color=UNLINKED_COLOR, width=0.6, dashes="[2 2] 0", oc=unlinked, overlay=True)
        if e["kind"] == "figure":
            flag_figure(doc, e, ids, k.get("layers", []), heights, flags)
    legend(doc, units)  # added last so page indices above stay right
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{units['paper']}_units.pdf"
    doc.save(str(out))
    doc.close()
    return out


def flag_figure(doc, fig: dict, asm_ids: list[str], layer_ids: list[str], heights: dict, oc: int) -> None:
    """A heavy box and a label naming the figure and what it links to."""
    for page, rect in rects(fig, heights):
        doc[page].draw_rect(rect, color=FLAG_COLOR, width=2.2, dashes="[6 3] 0", oc=oc, overlay=True)
        name = f"Fig {fig['figure_number']}" if fig.get("figure_number") else fig["id"]
        text = f"{name} -> {', '.join(asm_ids) or 'no assemblage'}" + (f" | {', '.join(layer_ids)}" if layer_ids else "")
        y = max(rect.y0 - 3, 9)
        doc[page].insert_text((rect.x0, y), text, fontsize=8, color=FLAG_COLOR, oc=oc, overlay=True)


def legend(doc, units: dict) -> None:
    """A summary page in front: counts, then each assemblage's colour, layers and details."""
    w, h = doc[0].rect.width, doc[0].rect.height
    page = doc.new_page(0, width=w, height=h)
    producer = units.get("producer", {})
    lines = [(units["paper"], 12), (f"{producer.get('tool', '')}  {producer.get('model', '')}  {producer.get('run_id', '')}", 7),
             (f"{units['counts']['n_assemblages']} assemblages, {units['counts']['n_layers']} layers", 11)]
    y = 40
    for text, size in lines:
        page.insert_text((36, y), text[:110], fontsize=size, color=FLAG_COLOR)
        y += size + 8
    for i, a in enumerate(units["assemblages"]):
        page.draw_rect(fitz.Rect(36, y - 8, 46, y + 1), color=colour(i), fill=colour(i))
        layers = ", ".join(l["name"] for l in units["layers"] if l["id"] in a.get("layer_ids", [])) or "no named layer"
        details = f"{a.get('locality') or ''} | {a.get('industry') or ''} | n={a.get('n_artefacts')}"
        page.insert_text((52, y), f"{a['id']}  {a['label']}"[:100], fontsize=9, color=FLAG_COLOR)
        page.insert_text((52, y + 11), f"layers: {layers}"[:120], fontsize=7, color=FLAG_COLOR)
        page.insert_text((52, y + 20), details[:120], fontsize=7, color=FLAG_COLOR)
        y += 34
    page.insert_text((36, y + 10), "Boxes: assemblage colour (nested when an element has several). "
                     "Dashed black: figure flag. Layers panel toggles each.", fontsize=7, color=FLAG_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path, help="the pdf_ocr corpus")
    ap.add_argument("--run", required=True, type=Path, help="a folder of *.paper_units.json")
    ap.add_argument("--only", help="substring of one paper folder name")
    ap.add_argument("--out-dir", type=Path, help="default: <run>/annotated")
    args = ap.parse_args()
    for f in sorted(args.run.glob("*.paper_units.json")):
        units = json.loads(f.read_text(encoding="utf-8"))
        if args.only and args.only not in units["paper"]:
            continue
        out = annotate(args.input_dir / units["paper"], units, args.out_dir or args.run / "annotated")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
