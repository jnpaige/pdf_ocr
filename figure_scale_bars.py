#!/usr/bin/env python3
"""
figure_scale_bars.py — find the scale bar on each figure plate and read it.

WHY THIS IS ITS OWN STAGE. Several technological modes turn on absolute size:
Mode D2 is a backed flake over 5cm and D4 the same piece under 5cm; Mode E1 is
an elongated core-tool over 10cm. A crop with no scale in it cannot be assigned
to any of them, so the bar has to travel with every piece handed to a model.
Detecting it inside figure_pieces.py did not work: the dilation radius chosen to
group an artefact's views is usually large enough to swallow the bar into a
neighbouring piece, so a bar was found only where it happened to sit alone.

WHAT WORKS INSTEAD. Find the printed value first, then the rule beside it.
Surya reads scale text off these plates reliably — '3cm', '6 cm', '5 cm',
'IFRAO 10 cm' — far better than any geometry test picks out the bar itself,
because a few characters on white is what a document OCR model is built for.
The graphic bar is then the wide, flat, nearly-empty component nearest that
text. Reading the value is the point: the bar's pixel length divided by its
stated length gives px_per_cm, so downstream size questions are answerable in
centimetres rather than merely visible.

OUTPUT  <doc>/figures/figure_scale_bars.json — one entry per plate that has one:
    file, scale_text, value, unit, value_cm, text_bbox, bar_bbox, px_per_cm

Usage:
    uv run python figure_scale_bars.py --input-dir "path/to/corpus"
    uv run python figure_scale_bars.py --input-dir ... --only Galotti
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

UPSAMPLE = 3               # Surya reads small isolated glyphs better enlarged
INK_THRESHOLD = 210
BAR_RADIUS = 2
BAND = 3.0                 # search this many text-heights above and below
RULE_MIN_WIDTH = 0.03      # of plate width
RULE_MAX_WIDTH = 0.60
RULE_MIN_ASPECT = 3.0
RULE_MAX_HEIGHT = 2.0      # of the text height; ticks stick out
RULE_MAX_OFFSET = 0.25     # rule centre within this fraction of plate width
                           # of the text centre

SCALE_TEXT = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[.,]\d+)?)\s*(cm|mm|m|in|inch(?:es)?)(?![\w])", re.I)
TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "inch": 2.54, "inches": 2.54}


def load_surya():
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    return RecognitionPredictor(FoundationPredictor()), DetectionPredictor()


def ink_of(path: Path) -> tuple[np.ndarray, int, int]:
    a = np.array(Image.open(path).convert("L"))
    mask = a < INK_THRESHOLD
    h, w = mask.shape
    return mask, w, h


def rule_near(mask: np.ndarray, tb: list[int]) -> tuple[dict | None, dict]:
    """Find the graphic rule that belongs to a piece of scale text.

    Searching for the rule on its own does not work. A scale rule is a solid
    line, so within its own tight bounding box it is nearly ALL ink — the
    opposite of the sparse, mostly-empty box it looks like only when a larger
    dilation has already merged it with its caption. Anchor on the text
    instead, which Surya finds reliably, and look in a band around it for a
    component that is wide and thinner than the digits beside it.

    Returns (rule_bbox or None, region) where region is the text and rule
    together — what gets composited under a piece crop.
    """
    h, w = mask.shape
    tl, tt, tr, tb_ = tb
    th = max(tb_ - tt, 1)
    tw = max(tr - tl, 1)
    y0, y1 = max(int(tt - BAND * th), 0), min(int(tb_ + BAND * th), h)
    band = mask[y0:y1, :].copy()
    # Blank the text itself. A rule printed close to its number merges with it
    # under any dilation, and the merged component is then taller than the
    # digits and fails every test meant to identify a rule.
    band[max(tt - y0, 0):max(tb_ - y0, 0), tl:tr] = False
    lab, _ = ndimage.label(ndimage.binary_dilation(
        band, ndimage.generate_binary_structure(2, 2), iterations=BAR_RADIUS))

    best = None
    for sl in ndimage.find_objects(lab):
        ys, xs = sl
        sub = band[sl]
        rows, cols = np.nonzero(sub)
        if rows.size == 0:
            continue
        l = int(xs.start + cols.min())
        r = int(xs.start + cols.max()) + 1
        t = y0 + int(ys.start + rows.min())
        b = y0 + int(ys.start + rows.max()) + 1
        bw, bh = r - l, b - t
        if bw < max(0.6 * tw, RULE_MIN_WIDTH * w) or bw > RULE_MAX_WIDTH * w:
            continue
        if bh > RULE_MAX_HEIGHT * th:    # ticks make it taller than the digits
            continue
        if bw < RULE_MIN_ASPECT * bh:
            continue
        if abs((l + r) / 2 - (tl + tr) / 2) > RULE_MAX_OFFSET * w:
            continue
        if best is None or bw > best["r"] - best["l"]:
            best = {"l": l, "t": t, "r": r, "b": b, "origin": "TOPLEFT"}

    parts = [(tl, tt, tr, tb_)] + ([(best["l"], best["t"], best["r"], best["b"])]
                                   if best else [])
    region = {"l": min(x[0] for x in parts), "t": min(x[1] for x in parts),
              "r": max(x[2] for x in parts), "b": max(x[3] for x in parts),
              "origin": "TOPLEFT"}
    return best, region


def read_plate(rec, det, path: Path) -> list[tuple[str, list[int]]]:
    im = Image.open(path).convert("RGB")
    up = im.resize((im.width * UPSAMPLE, im.height * UPSAMPLE), Image.LANCZOS)
    preds = rec([up], det_predictor=det)
    lines = preds[0].text_lines if preds else []
    return [((ln.text or "").strip(), [round(v / UPSAMPLE) for v in ln.bbox])
            for ln in lines if (ln.text or "").strip()]


def scale_for_plate(rec, det, path: Path) -> dict | None:
    """The best scale reading on this plate, or None.

    A plate may print its scale several times; prefer a reading whose rule was
    actually located, since only those yield px_per_cm.
    """
    mask, _w, _h = ink_of(path)
    best = None
    for text, tb in read_plate(rec, det, path):
        m = SCALE_TEXT.search(text)
        if not m:
            continue
        value, unit = float(m.group(1).replace(",", ".")), m.group(2).lower()
        cm = value * TO_CM.get(unit, TO_CM.get(unit[:2], 1.0))
        rule, region = rule_near(mask, tb)
        hit = {"scale_text": text, "value": value, "unit": unit,
               "value_cm": round(cm, 3),
               "text_bbox": {"l": tb[0], "t": tb[1], "r": tb[2], "b": tb[3],
                             "origin": "TOPLEFT"},
               "bar_bbox": rule,
               "region": region,
               "px_per_cm": round((rule["r"] - rule["l"]) / cm, 2)
                            if rule and cm else None}
        if hit["px_per_cm"] and (best is None or not best["px_per_cm"]):
            best = hit
        elif best is None:
            best = hit
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--only")
    ap.add_argument("--all-figures", action="store_true",
                    help="read every plate, not just those classified as "
                         "showing artefacts")
    args = ap.parse_args()

    # Same gate as figure_pieces.py — a scale bar is only wanted where there is
    # an artefact to measure against it, and reading every map and bar chart
    # would treble the Surya work for nothing.
    sys.path.insert(0, str(Path(__file__).parent))
    from figure_pieces import ARTIFACT_CLASSES, artifact_files, is_plate

    print("loading Surya...", flush=True)
    rec, det = load_surya()

    root = Path(args.input_dir)
    found = plates = 0
    with_px = [0]
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if args.only and args.only.lower() not in d.name.lower():
            continue
        fj = d / "figures" / "figures.json"
        if not fj.exists():
            continue
        data = json.loads(fj.read_text(encoding="utf-8"))
        figs = data.get("figures", []) if isinstance(data, dict) else data
        gate = None if args.all_figures else artifact_files(d, ARTIFACT_CLASSES)
        results = []
        for fig in figs:
            if not isinstance(fig, dict):
                continue
            p = d / "figures" / Path(fig.get("file") or "").name
            if not p.exists():
                continue
            if gate is not None and p.name not in gate:
                continue
            if not is_plate(p):
                continue
            plates += 1
            hit = scale_for_plate(rec, det, p)
            if hit:
                found += 1
                if hit["px_per_cm"]:
                    with_px[0] += 1
                results.append({"file": p.name, **hit})
                print(f"  {d.name[:26]:28s} {p.name:16s} "
                      f"{hit['scale_text'][:12]:14s} "
                      f"{str(hit['px_per_cm'] or 'rule not found'):>14s}", flush=True)
        if results:
            (d / "figures" / "figure_scale_bars.json").write_text(
                json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n{found} scale bars read across {plates} plates")


if __name__ == "__main__":
    main()
