#!/usr/bin/env python3
"""
figure_pieces.py — split an extracted figure plate into piece-sized regions.

Lithic plates carry many artefacts, often each drawn or photographed in two or
three views. The procedural unit and technological mode codebooks both ask
questions about individual pieces ("does this platform carry two or more
parallel facets"), so a figure-level label cannot answer them. This finds the
regions to crop.

Why not printed labels: tested on six plates, Surya recovers the printed piece
labels (a, b, c / 1, 2, 3) at about 40% recall — reliable on drawn plates where
the label sits in white space, near-useless on photographic plates where it is
printed over a dark image, because Surya detects lines of text in a document
layout and a single floating letter is not one. So grouping is purely
geometric, and a label is kept only when it happens to fall inside a group.

Method: binarize, then dilate the ink mask and take connected components. The
dilation radius sets how aggressively views merge into pieces, and is chosen
per plate by sweeping radii and taking the widest plateau in group count —
the scale at which the plate's structure is stable. Self-tuning, because this
has to run over hundreds of papers with no per-figure attention.

Usage:
    uv run python figure_pieces.py --input-dir "path/to/corpus"
    uv run python figure_pieces.py --input-dir ... --annotate   # write QA PNGs
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

INK_THRESHOLD = 210        # below this grey value counts as ink
MIN_GROUP_AREA = 0.004     # of plate area; drops specks and stray marks
CAPTION_MIN_WIDTH = 0.55   # a group this wide and this flat is caption text,
CAPTION_MAX_HEIGHT = 0.06  # not an artefact
SPARSE_INK = 0.11          # scale bars are a rule and a couple of digits: wide,
SPARSE_ASPECT = 2.5        # nearly empty, and they must not become pieces
BAR_MIN_AREA = 0.0004      # but a bar is still bigger than a speck
SPLIT_WIDE = 1.7           # a group this much wider than its peers is a row of
                           # pieces, and is re-labelled at a smaller radius
RADII = range(3, 41, 2)


def ink_mask(path: Path) -> tuple[np.ndarray, Image.Image]:
    im = Image.open(path).convert("L")
    return np.array(im) < INK_THRESHOLD, im


def groups_at(mask: np.ndarray, radius: int) -> tuple[np.ndarray, int]:
    grown = ndimage.binary_dilation(mask, ndimage.generate_binary_structure(2, 2),
                                    iterations=radius)
    return ndimage.label(grown)


UNITS = re.compile(r"^(cm|mm|m|km|kg|g|ka|ma|myr|yr|bp)\b", re.I)


def expected_pieces(caption: str) -> int | None:
    """How many pieces the caption says are on the plate.

    Captions enumerate their pieces — "1 e 4, 6, unifacial unidirectional
    cores", "Items 1-3, 6-8 are retouched", "(a-e): elongated blanks". That
    roster is the one thing a caption reliably carries (13% mention a
    procedural feature; the enumerations are far more common), so it is used
    here only to constrain grouping, never as evidence.
    """
    if not caption:
        return None
    body = re.sub(r"^\s*(?:extended\s+data\s+)?fig(?:ure)?\.?\s*\d+[.:|]?\s*", "",
                  caption.strip(), flags=re.I)
    nums, letters = [], []
    for m in re.finditer(r"(?<![\w.])\(?([a-r]|\d{1,2})\)?(?=\s*[-–—,;)]|\s*$)", body, re.I):
        tok, tail = m.group(1), body[m.end():m.end() + 4].lstrip()
        if UNITS.match(tail):
            continue
        if tok.isdigit():
            nums.append(int(tok))
        else:
            letters.append(ord(tok.lower()) - 96)
    best = max(nums + letters, default=0)
    return best if 2 <= best <= 40 else None


def pick_radius(mask: np.ndarray, expect: int | None) -> int:
    """Choose the dilation radius.

    With a caption roster, take the radius whose group count lands closest to
    it, preferring the larger radius on a tie so that multiple views of one
    piece merge rather than split. Without one, take the smallest radius at
    which the count has gone locally stable — a global widest-plateau search
    runs away to the over-merged end, where every plate is one blob.
    """
    counts = [(r, groups_at(mask, r)[1]) for r in RADII]
    if expect:
        return min(counts, key=lambda rc: (abs(rc[1] - expect), -rc[0]))[0]
    for i, (r, n) in enumerate(counts):
        ahead = [c for _, c in counts[i + 1:i + 3]]
        if n > 1 and ahead and all(abs(c - n) <= max(1, 0.1 * n) for c in ahead):
            return r
    return counts[len(counts) // 2][0]


def split_wide(mask: np.ndarray, g: dict, median_w: float, plate_w: int,
               radius: int, depth: int = 0) -> list[dict]:
    """A group far wider than its peers is a row of pieces the dilation joined.

    Splitting on a fixed whitespace width cannot work: dilation at radius r has
    already closed every gap narrower than 2r, so the surviving gaps inside the
    group are all narrower than the threshold that would identify them. Instead
    re-label the group's own ink at a smaller radius, which separates whatever
    needed less merging than its neighbours did.
    """
    width = g["r"] - g["l"]
    if width < SPLIT_WIDE * median_w or radius <= 2 or depth >= 1:
        return [g]
    sub = mask[g["t"]:g["b"], g["l"]:g["r"]]
    for r2 in (radius // 2, radius // 3, 1):
        r2 = max(r2, 1)
        lab2, n2 = groups_at(sub, r2)
        if n2 > 1:
            break
    else:
        return [g]
    if n2 <= 1:
        return [g]
    parts = []
    for sl in ndimage.find_objects(lab2):
        ys, xs = sl
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        if bw * bh < 0.05 * width * (g["b"] - g["t"]):
            continue
        parts.append({**g, "l": g["l"] + int(xs.start), "t": g["t"] + int(ys.start),
                      "r": g["l"] + int(xs.stop), "b": g["t"] + int(ys.stop),
                      "ink_px": int(sub[sl].sum())})
    if len(parts) <= 1:
        return [g]
    return [q for part in parts
            for q in split_wide(mask, part, median_w, plate_w, r2, depth + 1)]


def build(mask: np.ndarray, radius: int, w: int, h: int,
          picture_h: float | None = None) -> tuple[list[dict], list[dict]]:
    """Dilate, label, and separate artefacts from the plate's furniture.

    Returns (pieces, scale_bars). The scale bar is not discarded: several
    technological modes turn on absolute size — D2 is a backed flake over 5cm
    and D4 the same under 5cm, E1 an elongated tool over 10cm — so a crop with
    no scale in it cannot be assigned to any of them. It travels with each
    piece at the same pixel scale.
    """
    lab, _ = groups_at(mask, radius)
    out, bars = [], []
    for sl in ndimage.find_objects(lab):
        ys, xs = sl
        # Measure and crop on the group's own ink, not the dilated hull: the
        # hull grows with the radius, which both wastes crop margin and makes
        # a scale bar look squarer than it is, slipping past the aspect test.
        sub = mask[sl]
        rows, cols = np.nonzero(sub)
        if rows.size == 0:
            continue
        l, t = int(xs.start + cols.min()), int(ys.start + rows.min())
        r, b = int(xs.start + cols.max()) + 1, int(ys.start + rows.max()) + 1
        bw, bh = r - l, b - t
        ink = int(sub.sum())
        if picture_h and t >= picture_h:
            continue                                   # in the caption strip
        # Test for the scale bar BEFORE the artefact area floor: a bar is a
        # rule and two or three characters, routinely smaller than the
        # smallest artefact on the plate, so the area floor would discard it
        # before it could be recognised as anything.
        if (ink < SPARSE_INK * bw * bh and bw > SPARSE_ASPECT * bh
                and bw * bh > BAR_MIN_AREA * w * h):
            bars.append({"l": l, "t": t, "r": r, "b": b,
                         "ink_px": ink, "origin": "TOPLEFT"})
            continue                                   # scale bar
        if bw * bh < MIN_GROUP_AREA * w * h:
            continue
        if bw > CAPTION_MIN_WIDTH * w and bh < CAPTION_MAX_HEIGHT * h:
            continue                                   # caption line
        out.append({"l": l, "t": t, "r": r, "b": b,
                    "ink_px": ink, "origin": "TOPLEFT"})
    if out:
        median_w = float(np.median([g["r"] - g["l"] for g in out]))
        out = [q for g in out for q in split_wide(mask, g, median_w, w, radius)]
        # Re-apply after splitting: a group can start in the picture and reach
        # into the caption, and its lower parts are only separable once split.
        if picture_h:
            out = [g for g in out if g["t"] < picture_h]
    if picture_h:
        bars = [g for g in bars if g["t"] < picture_h]
    return out, bars


def pieces(path: Path, caption: str = "", bbox=None) -> tuple[list[dict], int, tuple[int, int], int | None]:
    """Regions to crop, one per artefact as far as geometry can tell.

    Radius is swept over the whole pipeline — dilation, filtering and splitting
    together — rather than chosen against an intermediate count, because the
    split stage changes the count and the two otherwise pull against each other.
    """
    mask, _im = ink_mask(path)
    h, w = mask.shape
    expect = expected_pieces(caption)

    # The exported PNG is usually the picture with a caption strip appended
    # below, so the plate is taller than the figure itself. A multi-line
    # caption is not flat enough for the caption-line filter to catch, and it
    # would otherwise become a "piece" and, once mapped back onto the page,
    # draw a box across the article's body text. The bbox is isotropic with
    # the render, so the picture's true height follows from its aspect.
    picture_h = None
    if bbox:
        bl, bt, br, bb_ = (bbox if isinstance(bbox, (list, tuple))
                           else (bbox["l"], bbox["t"], bbox["r"], bbox["b"]))
        bw_, bh_ = abs(br - bl), abs(bt - bb_)
        if bw_ and bh_:
            picture_h = min(w * (bh_ / bw_) * 1.02, h)

    results = [(r, build(mask, r, w, h, picture_h)) for r in RADII]
    counts = [(r, len(g)) for r, (g, _b) in results]
    if expect:
        radius = min(counts, key=lambda rc: (abs(rc[1] - expect), -rc[0]))[0]
    else:
        radius = counts[len(counts) // 2][0]
        for i, (r, n) in enumerate(counts):
            ahead = [c for _, c in counts[i + 1:i + 3]]
            if n > 1 and ahead and all(abs(c - n) <= max(1, 0.1 * n) for c in ahead):
                radius = r
                break
    out, _bars = dict(results)[radius]
    out.sort(key=lambda g: (g["t"] // max(h // 12, 1), g["l"]))   # reading order
    return out, radius, (w, h), expect


CROP_PAD = 8          # px of white around the artefact
BAR_GAP = 10          # px between the artefact and the scale bar beneath it


def export_crops(path: Path, groups: list[dict], scale: dict | None, dest_dir: Path,
                 stem: str) -> list[str]:
    """One PNG per piece, with the plate's scale bar composited beneath it.

    The bar is pasted at its original pixel size, unscaled and on the same
    canvas as the artefact, so the ratio between them is the plate's own and a
    model can read absolute size off the pair. Scaling either one would make
    the bar decorative and the size codes unanswerable.
    """
    im = Image.open(path).convert("RGB")
    bar_im = None
    if scale and scale.get("region"):
        b = scale["region"]
        bar_im = im.crop((max(b["l"] - 4, 0), max(b["t"] - 4, 0),
                          min(b["r"] + 4, im.width), min(b["b"] + 4, im.height)))
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, g in enumerate(groups, start=1):
        crop = im.crop((max(g["l"] - CROP_PAD, 0), max(g["t"] - CROP_PAD, 0),
                        min(g["r"] + CROP_PAD, im.width), min(g["b"] + CROP_PAD, im.height)))
        if bar_im is not None:
            w = max(crop.width, bar_im.width)
            h = crop.height + BAR_GAP + bar_im.height
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(crop, (0, 0))
            canvas.paste(bar_im, (0, crop.height + BAR_GAP))
            crop = canvas
        name = f"{stem}_piece{i:02d}.png"
        crop.save(dest_dir / name)
        written.append(name)
    return written


def annotate(path: Path, groups: list[dict], dest: Path) -> None:
    im = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for i, g in enumerate(groups, start=1):
        dr.rectangle([g["l"], g["t"], g["r"], g["b"]], outline=(220, 30, 30), width=3)
        dr.text((g["l"] + 4, g["t"] + 2), str(i), fill=(220, 30, 30))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)


ARTIFACT_CLASSES = ("lithic_tool", "artifact_image")
MIN_PLATE_PX = 200         # below this on either side it is page furniture


def is_plate(path: Path) -> bool:
    """Is this extracted picture big enough to be a figure at all?

    The classifier cannot be relied on here. Broich's running-header logo is
    79x24 pixels, repeats on eleven pages, and qwen2.5vl:7b labels every copy
    `artifact_image` — 37% of everything the class gate admitted was furniture
    of this kind. A size floor is the honest test: nothing 79 pixels wide is a
    plate of artefacts, whatever a vision model calls it.
    """
    try:
        with Image.open(path) as im:
            return min(im.size) >= MIN_PLATE_PX
    except Exception:
        return False


def artifact_files(doc_dir: Path, wanted: tuple[str, ...]) -> set[str] | None:
    """Which plates in this document actually show artefacts.

    Reads classify_figures.py's per-document output, whose taxonomy has
    lithic_tool and artifact_image and is multi-label. Docling's own
    `classification` field is NOT usable for this: on lithic papers it files
    artefact photographs under `icon` and `logo` — Araho's "Nearly complete
    example of a Type 1 stemmed tool" comes back as `icon` — and it spreads
    real plates across five classes. Returns None when no classification has
    been run, meaning "no gate", so the script still works standalone.
    """
    runs = doc_dir.parent / "000_segments" / "runs"
    if not runs.is_dir():
        return None
    keep: set[str] = set()
    seen = False
    for run in sorted(runs.iterdir(), reverse=True):
        hits = list(run.glob(f"*{doc_dir.name}.figure_segments.json"))
        if not hits:
            continue
        seen = True
        data = json.loads(hits[0].read_text(encoding="utf-8"))
        for fig in data.get("figures", []):
            if set(fig.get("classes") or []) & set(wanted):
                keep.add(Path(str(fig.get("file") or "")).name)
        break
    return keep if seen else None


def scale_bar_for(doc_dir: Path) -> dict:
    """Per-plate scale readings from figure_scale_bars.py, keyed by file."""
    f = doc_dir / "figures" / "figure_scale_bars.json"
    if not f.exists():
        return {}
    try:
        return {e["file"]: e for e in json.loads(f.read_text(encoding="utf-8"))}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--crops", action="store_true",
                    help="write one PNG per piece, scale bar composited beneath")
    ap.add_argument("--only", help="restrict to document folders containing this string")
    ap.add_argument("--all-figures", action="store_true",
                    help="process every plate, not just those classified as "
                         "showing artefacts")
    args = ap.parse_args()

    root = Path(args.input_dir)
    print(f"{'document / figure':38s} {'plate':>11s} {'rad':>4s} {'capt':>6s} {'found':>7s} {'bar':>4s}")
    grand = 0
    with_bar = [0]
    skipped = [0]
    furniture = [0]
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if args.only and args.only.lower() not in d.name.lower():
            continue
        fj = d / "figures" / "figures.json"
        if not fj.exists():
            continue
        data = json.loads(fj.read_text(encoding="utf-8"))
        figs = data.get("figures", []) if isinstance(data, dict) else data
        gate = None if args.all_figures else artifact_files(d, ARTIFACT_CLASSES)
        bars = scale_bar_for(d)
        if gate is not None:
            skipped[0] += sum(1 for f in figs if isinstance(f, dict)
                              and Path(str(f.get("file") or "")).name not in gate)
        manifest = []
        for fig in figs:
            if not isinstance(fig, dict):
                continue
            p = d / "figures" / Path(fig.get("file") or "").name
            if not p.exists():
                continue
            if gate is not None and p.name not in gate:
                continue
            if not is_plate(p):
                furniture[0] += 1
                continue
            groups, radius, (w, h), expect = pieces(
                p, fig.get("caption") or "", fig.get("bbox"))
            manifest.append({"file": p.name, "page": fig.get("page"),
                             "printed_figure_number": fig.get("printed_figure_number"),
                             "plate_px": [w, h], "dilation_radius": radius,
                             "caption_expected_pieces": expect,
                             "pieces": groups})
            grand += len(groups)
            print(f"{d.name[:24]:26s} fig {str(fig.get('printed_figure_number')):5s} "
                  f"{w:5d}x{h:<5d} {radius:4d} {str(expect or '-'):>6s} {len(groups):7d} "
                  f"{('yes' if bars.get(p.name) else '-'):>4s}")
            if args.annotate:
                annotate(p, groups, d / "figures" / "pieces_qa" / f"{p.stem}_pieces.png")
            if args.crops:
                names = export_crops(p, groups, bars.get(p.name),
                                     d / "figures" / "piece_crops", p.stem)
                manifest[-1]["crops"] = names
                if bars.get(p.name):
                    with_bar[0] += len(names)
        if manifest:
            (d / "figures" / "figure_pieces.json").write_text(
                json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\n{grand} piece regions")


if __name__ == "__main__":
    main()
