"""Run Surya OCR on a list of page images and return structured results.

Supports three pipeline modes (set via surya.pipeline in config.yaml):

  recognition   — Original behaviour. Runs DetectionPredictor + RecognitionPredictor
                   directly on the full page image.  Fast, but no column/layout
                   awareness: text from adjacent columns or figures can merge.

  ocr           — Uses Surya's high-level OCRPredictor, which internally chains
                   LayoutPredictor → DetectionPredictor → RecognitionPredictor.
                   Layout analysis identifies columns, figures, and tables as
                   separate regions and OCRs each independently.

  layout_rec    — Runs LayoutPredictor explicitly, then feeds each detected text
                   region to RecognitionPredictor as a separate cropped image.
                   Before OCR'ing each text region, the crop is clipped to exclude
                   any overlapping figure/picture areas, preventing figure text
                   from bleeding into column results.  Gives full control over
                   which region types are OCR'd via layout_region_types.
"""
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # large scanned maps exceed PIL's default safety limit


# ---------------------------------------------------------------------------
# Pipeline: recognition  (original)
# ---------------------------------------------------------------------------
def _run_recognition(image_paths: list[Path], languages: list[str]) -> list[dict]:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

    print("  Loading Surya models (recognition pipeline)...")
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"  Running OCR on {len(images)} page(s)...")
    predictions = rec_predictor(images, det_predictor=det_predictor)

    return _predictions_to_results(predictions, image_paths)


# ---------------------------------------------------------------------------
# Pipeline: ocr  (layout-aware, high-level)
# ---------------------------------------------------------------------------
def _run_ocr_predictor(image_paths: list[Path], languages: list[str]) -> list[dict]:
    from surya.ocr import OCRPredictor

    print("  Loading Surya models (ocr pipeline — layout-aware)...")
    ocr_predictor = OCRPredictor()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"  Running layout-aware OCR on {len(images)} page(s)...")
    predictions = ocr_predictor(images)

    return _predictions_to_results(predictions, image_paths)


# ---------------------------------------------------------------------------
# Pipeline: layout_rec  (explicit layout → per-region recognition)
# ---------------------------------------------------------------------------
def _run_layout_rec(
    image_paths: list[Path],
    languages: list[str],
    layout_region_types: list[str] | None = None,
) -> list[dict]:
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

    print("  Loading Surya models (layout_rec pipeline)...")
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    layout_predictor = LayoutPredictor(foundation_predictor)
    rec_predictor = RecognitionPredictor(foundation_predictor)

    if layout_region_types is None:
        layout_region_types = ["Text", "SectionHeader", "Title", "Caption",
                               "Footnote", "Formula", "ListItem",
                               "PageHeader", "PageFooter",
                               "Table", "Figure", "Picture"]

    images = [Image.open(p).convert("RGB") for p in image_paths]

    print(f"  Running layout analysis on {len(images)} page(s)...")
    layout_preds = layout_predictor(images)

    # Figure/Picture bboxes are collected as exclusion zones regardless of
    # whether they appear in layout_region_types.  Text crops are clipped
    # to these zones before OCR so figure text cannot bleed into column results.
    FIGURE_LABELS = {"Figure", "Picture"}

    results = []
    for page_idx, (layout, img, img_path) in enumerate(
        zip(layout_preds, images, image_paths)
    ):
        page_lines = []
        region_meta = []
        w, h = img.size

        regions = sorted(
            layout.bboxes,
            key=lambda r: (r.bbox[1], r.bbox[0]),
        )

        figure_zones = [
            [int(v) for v in r.bbox]
            for r in regions
            if (r.label if hasattr(r, "label") else "") in FIGURE_LABELS
        ]

        for region in regions:
            label = region.label if hasattr(region, "label") else "Unknown"
            region_meta.append({
                "label":      label,
                "bbox":       region.bbox,
                "confidence": round(region.confidence, 4)
                              if hasattr(region, "confidence") else None,
            })

            if label not in layout_region_types:
                continue

            x0, y0, x1, y1 = [int(v) for v in region.bbox]

            # Clip non-figure crops to exclude figure areas before OCR.
            # This prevents figure labels/annotations from being picked up
            # when a Text region bbox extends into an adjacent figure.
            if label not in FIGURE_LABELS and figure_zones:
                x0, y0, x1, y1 = _clip_to_exclude_figures(
                    x0, y0, x1, y1, figure_zones
                )

            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(w, x1); y1 = min(h, y1)
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue

            crop = img.crop((x0, y0, x1, y1))
            preds = rec_predictor([crop], det_predictor=det_predictor)
            if not preds:
                continue

            for line in preds[0].text_lines:
                lb = line.bbox
                page_lines.append({
                    "text":       line.text,
                    "confidence": round(line.confidence, 4),
                    "bbox":       [lb[0] + x0, lb[1] + y0,
                                   lb[2] + x0, lb[3] + y0],
                    "region":     label,
                })

        results.append({
            "page_index":     page_idx,
            "image_path":     str(img_path),
            "text_lines":     page_lines,
            "full_text":      "\n".join(l["text"] for l in page_lines),
            "layout_regions": region_meta,
        })

    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clip_to_exclude_figures(
    x0: int, y0: int, x1: int, y1: int,
    figure_zones: list[list[int]],
) -> tuple[int, int, int, int]:
    """Shrink a text-region bbox to exclude overlapping figure areas.

    For each figure zone that overlaps, determines which side of the text
    region the figure is primarily on and clips that edge to the figure
    boundary.  Handles the common layout: figure to the left, right, above,
    or below a text column.
    """
    for fz in figure_zones:
        fx0, fy0, fx1, fy1 = fz
        if fx1 <= x0 or fx0 >= x1 or fy1 <= y0 or fy0 >= y1:
            continue  # no overlap

        cx  = (x0  + x1)  / 2
        cy  = (y0  + y1)  / 2
        fcx = (fx0 + fx1) / 2
        fcy = (fy0 + fy1) / 2

        if abs(fcx - cx) >= abs(fcy - cy):
            if fcx > cx:   # figure is to the right → pull right edge left
                x1 = min(x1, fx0)
            else:          # figure is to the left  → push left edge right
                x0 = max(x0, fx1)
        else:
            if fcy > cy:   # figure is below → pull bottom edge up
                y1 = min(y1, fy0)
            else:          # figure is above → push top edge down
                y0 = max(y0, fy1)

    return x0, y0, x1, y1


def _predictions_to_results(predictions, image_paths: list[Path]) -> list[dict]:
    """Convert Surya prediction objects to the standard result-dict format."""
    results = []
    for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
        text_lines = []
        for line in pred.text_lines:
            text_lines.append({
                "text":       line.text,
                "confidence": round(line.confidence, 4),
                "bbox":       line.bbox,
            })
        results.append({
            "page_index": i,
            "image_path": str(img_path),
            "text_lines": text_lines,
            "full_text":  "\n".join(l["text"] for l in text_lines),
        })
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_ocr(
    image_paths: list[Path],
    languages: list[str],
    surya_cfg: dict | None = None,
) -> list[dict]:
    """OCR each image with Surya and return one result dict per page.

    Parameters
    ----------
    image_paths : list[Path]
        Rendered page images.
    languages : list[str]
        Language codes (e.g. ["en"]).
    surya_cfg : dict, optional
        The 'surya' block from config.yaml.  Recognised keys:
          pipeline             : "recognition" | "ocr" | "layout_rec"
          layout_region_types  : list[str]  (layout_rec only)

    Returns a list of dicts with keys:
      page_index, image_path, text_lines, full_text
    """
    if surya_cfg is None:
        surya_cfg = {}

    pipeline = surya_cfg.get("pipeline", "recognition")

    if pipeline == "recognition":
        return _run_recognition(image_paths, languages)

    elif pipeline == "ocr":
        return _run_ocr_predictor(image_paths, languages)

    elif pipeline == "layout_rec":
        region_types = surya_cfg.get("layout_region_types", None)
        return _run_layout_rec(image_paths, languages, region_types)

    else:
        raise ValueError(
            f"Unknown surya pipeline '{pipeline}'. "
            f"Choose from: recognition, ocr, layout_rec"
        )