"""Docling pipeline with Surya as the OCR backend.

Implements SuryaOcrModel (satisfies docling's BaseOcrModel interface) and
SuryaPdfPipeline (overrides _make_ocr_model to bypass the factory and inject
the Surya model directly). Docling handles layout, tables, and reading-order;
Surya handles all OCR including handwriting.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Optional, Type

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # large scanned pages exceed PIL's default safety limit


# ---------------------------------------------------------------------------
# Surya OCR options  (thin Pydantic wrapper so BaseOcrModel is satisfied)
# ---------------------------------------------------------------------------
from docling.datamodel.pipeline_options import OcrOptions


class SuryaOcrOptions(OcrOptions):
    kind: ClassVar[Literal["surya"]] = "surya"
    lang: list[str] = ["en"]  # OcrOptions requires lang; Surya ignores it (auto-detects)


# ---------------------------------------------------------------------------
# Surya OCR model — plugs into docling's BaseOcrModel interface
# ---------------------------------------------------------------------------
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder


# Raw per-page OCR text, captured straight from Surya before docling's layout
# model gets a chance to misclassify a page (e.g. a page of dense handwriting)
# as a Picture and drop all its text cells from the assembled document.
# Keyed by 1-indexed page_no; cleared at the start of each run_ocr() call.
# Module-level because DocumentConverter owns the pipeline/model lifecycle, so
# run_ocr() has no direct handle to the SuryaOcrModel instance to read from
# afterward. Safe because PDFs are processed one at a time, single-process.
_raw_ocr_cells_by_page: dict[int, list[TextCell]] = {}


def _cells_reading_order_text(cells: list[TextCell]) -> str:
    """Join OCR cells into text approximating reading order (top-to-bottom,
    left-to-right within a line band)."""
    def sort_key(c: TextCell):
        bbox = c.rect.to_bounding_box()
        return (round(bbox.t / 15.0), bbox.l)

    ordered = sorted(cells, key=sort_key)
    return "\n".join(c.text for c in ordered if c.text and c.text.strip())


BBOX_ORIGIN = "BOTTOMLEFT"


def _bbox_to_dict(bbox, page_height: float | None = None) -> dict:
    """Serialize a docling BoundingBox as a plain dict, normalized to a
    BOTTOMLEFT origin.

    Docling hands out mixed conventions — text-item provenance is BOTTOMLEFT,
    table cell bboxes are TOPLEFT, raw Surya cells follow whatever rectangle
    they came from. A consumer that assumes one convention draws every box
    from the other mirrored vertically about the page, which looks like a
    plausible box in the wrong place rather than an error. Converting once,
    here, means a bbox in this output means the same thing no matter which
    part of docling produced it.

    `origin` is still written on every box and consumers should still branch
    on it: it is the contract, it keeps already-written output (all
    BOTTOMLEFT) readable by the same code, and a box that could not be
    converted must still describe itself honestly.

    page_height is needed to flip a TOPLEFT box. Without it the box is passed
    through carrying its true origin rather than being silently mislabelled
    as normalized.
    """
    if bbox is None:
        return {}
    origin = getattr(bbox, "coord_origin", None)
    origin_value = origin.value if origin is not None else "TOPLEFT"
    if origin_value != BBOX_ORIGIN and page_height:
        try:
            bbox = bbox.to_bottom_left_origin(page_height)
            origin_value = BBOX_ORIGIN
        except Exception:
            pass
    return {
        "l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b,
        "origin": origin_value,
    }


def _page_heights(doc) -> dict[int, float]:
    """{0-indexed page -> height}, for normalizing bbox coordinate origins."""
    heights: dict[int, float] = {}
    for page_no, page in (getattr(doc, "pages", None) or {}).items():
        size = getattr(page, "size", None)
        height = getattr(size, "height", None)
        if height:
            heights[page_no - 1] = float(height)
    return heights


def _prov_bboxes(prov, page_heights: dict[int, float]) -> list[dict]:
    """Every region a docling item occupies, each tagged with its own page.

    An item's `prov` is a list, and a paragraph that flows across a column
    break — or across a page break — records one entry per region. Reading
    only `prov[0]` silently truncates such an item's geometry to its first
    fragment: a 1167-word paragraph spanning three regions reported a box
    covering roughly a fifth of itself, which looks like a correctly drawn
    box around the wrong amount of text rather than like an error.

    The page number travels with each region because later regions routinely
    fall on the *following* page, while the item's text stays attached to the
    page its first region is on.
    """
    regions: list[dict] = []
    for entry in (prov or []):
        page_index = entry.page_no - 1
        box = _bbox_to_dict(entry.bbox, page_heights.get(page_index))
        if box:
            box["page"] = page_index
            regions.append(box)
    return regions


def _union_bboxes(boxes: list[dict]) -> dict:
    """Union of already-normalized BOTTOMLEFT boxes (t is the upper edge)."""
    usable = [b for b in boxes if b and b.get("origin") == BBOX_ORIGIN]
    if not usable:
        return {}
    return {
        "l": min(b["l"] for b in usable), "t": max(b["t"] for b in usable),
        "r": max(b["r"] for b in usable), "b": min(b["b"] for b in usable),
        "origin": BBOX_ORIGIN,
    }


def _table_to_dict(item, page_heights: dict[int, float]) -> dict:
    """Serialize a docling TableItem: its region, its grid, and geometry per
    cell and per row.

    Tables used to leave this output with no coordinates at all. TableItem has
    no `.text` attribute, so the text-item loop below skipped it before ever
    reading its provenance, and a table survived only as pipe rows in md_text.
    Anything wanting to point at a table row on the page — an overlay, a QA
    highlighter, a per-row tag — had nothing to point at, even though docling
    had computed the geometry and was holding it.

    Row rectangles are unioned from the cells sharing a row index, so a
    consumer can address a row directly instead of reconstructing it.
    """
    prov = getattr(item, "prov", None)
    data = getattr(item, "data", None)
    page_height = page_heights.get((prov[0].page_no - 1) if prov else 0)

    cells: list[dict] = []
    row_boxes: dict[int, list[dict]] = defaultdict(list)
    row_is_header: dict[int, bool] = defaultdict(bool)

    for cell in (getattr(data, "table_cells", None) or []):
        bbox = _bbox_to_dict(getattr(cell, "bbox", None), page_height)
        start_row = cell.start_row_offset_idx
        end_row = max(cell.end_row_offset_idx, start_row + 1)
        cells.append({
            "text": cell.text,
            "bbox": bbox,
            "start_row": start_row, "end_row": cell.end_row_offset_idx,
            "start_col": cell.start_col_offset_idx, "end_col": cell.end_col_offset_idx,
            "column_header": bool(cell.column_header),
            "row_header": bool(cell.row_header),
        })
        for r in range(start_row, end_row):
            if bbox:
                row_boxes[r].append(bbox)
            row_is_header[r] = row_is_header[r] or bool(cell.column_header)

    return {
        "bbox": _bbox_to_dict(prov[0].bbox, page_height) if prov else {},
        "bboxes": _prov_bboxes(prov, page_heights),
        "num_rows": getattr(data, "num_rows", 0),
        "num_cols": getattr(data, "num_cols", 0),
        "rows": [
            {"row": r, "column_header": row_is_header[r], "bbox": _union_bboxes(row_boxes[r])}
            for r in sorted(row_boxes)
        ],
        "cells": cells,
    }


class SuryaOcrModel(BaseOcrModel):
    scale = 2  # render at 144 dpi (72 * 2); sufficient for Surya

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        if self.enabled:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            print("  Loading Surya models...")
            _foundation = FoundationPredictor()
            self._det = DetectionPredictor()
            self._rec = RecognitionPredictor(_foundation)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    img = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    preds = self._rec([img], det_predictor=self._det)
                    del img

                    if not preds:
                        continue

                    # line.bbox is [x0, y0, x1, y1] in crop-image pixels.
                    # Divide by scale to get page points, then offset by ocr_rect origin.
                    for ix, line in enumerate(preds[0].text_lines):
                        lb = line.bbox
                        all_ocr_cells.append(
                            TextCell(
                                index=ix,
                                text=line.text,
                                orig=line.text,
                                from_ocr=True,
                                confidence=round(line.confidence, 4),
                                rect=BoundingRectangle.from_bounding_box(
                                    BoundingBox.from_tuple(
                                        coord=(
                                            lb[0] / self.scale + ocr_rect.l,
                                            lb[1] / self.scale + ocr_rect.t,
                                            lb[2] / self.scale + ocr_rect.l,
                                            lb[3] / self.scale + ocr_rect.t,
                                        ),
                                        origin=CoordOrigin.TOPLEFT,
                                    )
                                ),
                            )
                        )

                self.post_process_cells(all_ocr_cells, page)
                _raw_ocr_cells_by_page[page.page_no] = all_ocr_cells

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return SuryaOcrOptions


# ---------------------------------------------------------------------------
# Custom pipeline — bypasses the OCR factory to inject SuryaOcrModel directly
# ---------------------------------------------------------------------------
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class SuryaPdfPipeline(StandardPdfPipeline):
    def _make_ocr_model(self, art_path: Optional[Path]):
        return SuryaOcrModel(
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=art_path,
            options=SuryaOcrOptions(),
            accelerator_options=self.pipeline_options.accelerator_options,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_REMOTE_PICTURE_DESCRIPTION_KEYS = (
    "picture_description_model",
    "picture_description_base_url",
    "picture_description_prompt",
    "picture_description_timeout",
)


def _reject_remote_picture_description(docling_cfg: dict) -> None:
    """Fail on any leftover picture-description config key.

    Docling's picture-description stage POSTs each extracted figure image to
    a configurable HTTP endpoint. To help meet broad data security
    requirements across domains, that path is walled off here rather than
    left wired up behind a local-looking default — a default is only a
    default, and the key that overrides it gives no sign of what changes
    when it does.

    Figure captioning is done instead by the separate figure_captioner pass,
    which runs a local model over already-extracted figure files.

    Fails on the keys rather than ignoring them, so an older config surfaces
    as an error at startup instead of quietly producing uncaptioned output
    that looks like a model problem.
    """
    present = [k for k in _REMOTE_PICTURE_DESCRIPTION_KEYS if k in docling_cfg]
    if present:
        raise ValueError(
            "Unsupported docling config key(s): " + ", ".join(present) + "\n"
            "Docling's built-in picture description sends figure image data to "
            "an outside server, so it is walled off in this pipeline. Remove "
            "these keys; caption figures with the figure_captioner repo, which "
            "uses a model running on this machine."
        )


def build_converter(docling_cfg: dict | None = None):
    """Build a Docling DocumentConverter configured for the Surya OCR pipeline.

    Expensive: constructing the pipeline loads Surya's OCR models
    (FoundationPredictor, DetectionPredictor, RecognitionPredictor — real
    transformer weights onto the GPU) plus Docling's own layout/table models.
    Build ONE of these per run.py invocation and pass it into every run_ocr()
    call for that run via the converter= param, rather than letting run_ocr()
    build its own — a fresh DocumentConverter per PDF (the previous default)
    reloads all of those models from scratch for every single document, which
    across a few thousand PDFs in one long-lived process is what caused
    creeping memory growth/OOM on large corpus runs, on top of burning most
    of the wall-clock time on reload rather than OCR.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    if docling_cfg is None:
        docling_cfg = {}

    _reject_remote_picture_description(docling_cfg)

    do_ocr = docling_cfg.get("do_ocr", True)
    extract_figures = docling_cfg.get("extract_figures", False)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr

    # Left at its default (False). Docling gates every pipeline stage that
    # calls out of the process behind this one flag, so it stays off here and
    # nothing in this repo turns it on — see
    # _reject_remote_picture_description.
    pipeline_options.enable_remote_services = False

    if extract_figures:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = docling_cfg.get("figures_scale", 2.0)
        pipeline_options.do_picture_classification = docling_cfg.get("classify_figures", False)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=SuryaPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def run_ocr(
    pdf_path: Path,
    out_dir: Path,
    docling_cfg: dict | None = None,
    stem: str | None = None,
    converter=None,
    stamp_page_labels: bool = True,
) -> list[dict]:
    """Run Docling + Surya on a PDF. Saves <stem>.md to out_dir.

    stem overrides pdf_path.stem for output file naming — used in backfill mode
    where the input is a *_ocr.pdf but outputs should use the canonical report name.

    converter: an optional pre-built DocumentConverter (see build_converter).
    Reused across many run_ocr() calls in one process instead of rebuilt per
    call — pass the same instance across a whole corpus run. Only built
    on-demand here (once) if omitted, for callers that just want one PDF done.

    stamp_page_labels: set False when this call is producing one chunk of a
    larger document that will be merged later (see run.py's
    process_pdf_chunked) — a chunk's own page 0 is NOT the merged document's
    page 0, so stamping here would bake in the wrong, chunk-local number.
    The merge step stamps the correct global page index once, after all
    chunks are concatenated, instead.

    Returns per-page result dicts with keys:
      page_index, image_path, text_lines, tables, full_text, md_text
    """
    if docling_cfg is None:
        docling_cfg = {}
    if stem is None:
        stem = pdf_path.stem

    _reject_remote_picture_description(docling_cfg)

    do_ocr = docling_cfg.get("do_ocr", True)
    extract_figures = docling_cfg.get("extract_figures", False)

    if converter is None:
        converter = build_converter(docling_cfg)

    _raw_ocr_cells_by_page.clear()

    if do_ocr:
        print(f"  Running Docling + Surya OCR...")
    else:
        print(f"  Running Docling (using existing text layer, do_ocr: false)...")
    result = converter.convert(pdf_path)
    doc = result.document

    results, n_recovered, fallback_pages = _build_page_results(doc, raw_cells=_raw_ocr_cells_by_page)
    if n_recovered:
        print(f"  Recovered {n_recovered} page(s) via raw OCR fallback "
              f"(layout model produced little/no text there)")

    # Page-delimited, same `=== Page N ===` convention as text_docling.txt,
    # with the same fallback substitution — see _build_page_results' docstring.
    md_content = "\n\n".join(f"=== Page {r['page_index']} ===\n{r['md_text']}" for r in results)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Saved markdown → {md_path.name}")

    flat_md_dir = docling_cfg.get("markdown_dir")
    if flat_md_dir:
        flat_dir = Path(flat_md_dir)
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_path = flat_dir / f"{stem}.md"
        flat_path.write_text(md_content, encoding="utf-8")
        print(f"  Mirrored markdown → {flat_path}")

    pdf_out = out_dir / f"{stem}_ocr.pdf"
    if do_ocr:
        _build_searchable_pdf(pdf_path, doc, pdf_out,
                               raw_cells=_raw_ocr_cells_by_page, fallback_pages=fallback_pages,
                               stamp_page_labels=stamp_page_labels)
        print(f"  Saved searchable PDF → {pdf_out.name}")
    else:
        # do_ocr: false means pdf_path already carries its own text layer (a
        # prior run's *_ocr.pdf, in from_ocr_pdf backfill mode) — there's no
        # fresh OCR pass to overlay, so _build_searchable_pdf doesn't apply.
        # Still stamp the `=== Page N ===` label so backfilled documents end
        # up matching fresh ones instead of permanently missing it (unless
        # this is one chunk of a larger document — see stamp_page_labels).
        _stamp_existing_pdf(pdf_path, pdf_out, stamp=stamp_page_labels)
        print(f"  Stamped page labels → {pdf_out.name} (do_ocr: false — reused existing text layer)")

    # Write headings.json alongside the other outputs
    txt_path = out_dir / "text_docling.txt"
    if txt_path.exists():
        h_path = write_headings_json(txt_path, out_dir, pdf_path.stem)
        print(f"  Saved headings    → {h_path.name}  ({_count_headings(h_path)} headings)")

    if extract_figures:
        n_figs = _extract_figures(doc, out_dir)
        print(f"  Saved figures     → figures/  ({n_figs} figure(s))")

    return results


def _extract_figures(doc, out_dir: Path) -> int:
    """Save each detected picture/figure as a PNG under <out_dir>/figures/, plus
    a figures.json manifest recording page, bbox, and classification (if
    enabled). Returns the count of figures saved.

    No caption field: figure_captioner is the captioning pass, and it writes
    its own <model_slug>__<figure_stem>.caption.json next to these files
    rather than editing this manifest, so nothing downstream reads a caption
    from here.
    """
    import json

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    manifest: list[dict] = []
    for i, pic in enumerate(doc.pictures):
        image = pic.get_image(doc)
        if image is None:
            continue

        prov = pic.prov[0] if pic.prov else None
        page_no = (prov.page_no - 1) if prov else None  # 0-indexed, matches text_docling.txt
        bbox = prov.bbox if prov else None

        fname = f"fig_p{page_no if page_no is not None else 'x'}_{i:03d}.png"
        image.save(figures_dir / fname)

        entry = {
            "file": fname,
            "page": page_no,
            "bbox": [bbox.l, bbox.t, bbox.r, bbox.b] if bbox else None,
        }

        classification = getattr(pic.meta, "classification", None) if pic.meta else None
        if classification is not None:
            pred = classification.get_main_prediction()
            entry["classification"] = pred.class_name

        manifest.append(entry)

    (figures_dir / "figures.json").write_text(
        json.dumps({"n_figures": len(manifest), "figures": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(manifest)


def _count_headings(h_path: Path) -> int:
    import json
    try:
        return json.loads(h_path.read_text(encoding="utf-8")).get("n_headings", 0)
    except Exception:
        return 0


def _stamp_page_label(fitz_page, page_idx: int) -> None:
    """Stamp `=== Page N ===` as real visible+selectable text top-right of
    one page — the same marker text_docling.txt/.md use, so a page in the
    PDF can be matched back to its .md/text_docling.txt section by eye or by
    copy-pasting straight out of a PDF viewer. Idempotent: skips pages that
    already carry the label (re-running on an already-stamped PDF, or a page
    _build_searchable_pdf already handled, is then a no-op here).

    Landscape tables in reports are commonly stored as a portrait page with
    a /Rotate 90 flag rather than actually-landscape page geometry.
    insert_text positions and draws glyphs in the page's raw (unrotated)
    coordinate system — it does not know about /Rotate — while `page.rect`
    reports the as-displayed (rotated) width/height. Using rect.width/height
    directly here silently placed the stamp entirely outside the visible
    page on any rotated page (found no exception, no stamp — just missing).
    derotation_matrix converts our desired on-screen top-right point into
    the correct raw coordinates, and rotate=page.rotation draws the glyphs
    so they still read upright in the rotated display. Both are identity /
    no-ops on the common unrotated case, so this is unconditionally safe.
    """
    import fitz

    label = f"=== Page {page_idx} ==="
    if label in fitz_page.get_text():
        return
    label_fontsize = 9
    label_width = fitz.get_text_length(label, fontsize=label_fontsize)
    pt_display = fitz.Point(fitz_page.rect.width - label_width - 15, 25)
    pt_raw = pt_display * fitz_page.derotation_matrix
    try:
        fitz_page.insert_text(
            pt_raw, label, fontsize=label_fontsize, color=(0.2, 0.2, 0.2),
            overlay=True, rotate=fitz_page.rotation,
        )
    except Exception:
        pass


def _stamp_existing_pdf(pdf_path: Path, out_path: Path, stamp: bool = True) -> None:
    """Stamp page-number labels onto a PDF that already has its own text
    layer (do_ocr: false — no fresh OCR pass to overlay, so
    _build_searchable_pdf's per-item text placement doesn't apply here, only
    the page label). Used for from_ocr_pdf backfill runs so an old *_ocr.pdf
    ends up with the same top-right `=== Page N ===` marker a fresh run
    would give it, without re-running Surya.

    stamp=False still copies pdf_path to out_path but skips labeling — used
    when this is one chunk of a larger document (see run_ocr's
    stamp_page_labels), where a chunk's own local page 0 isn't the merged
    document's page 0.

    pdf_path and out_path are commonly the same file (backfilling in place)
    — save to a sibling temp file and swap it in, since fitz can't safely
    save a document over the same path it was opened from.
    """
    import fitz

    src = fitz.open(str(pdf_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    same_path = Path(pdf_path).resolve() == out_path.resolve()
    save_path = out_path.with_suffix(out_path.suffix + ".tmp") if same_path else out_path

    if stamp:
        for page_idx in range(len(src)):
            _stamp_page_label(src[page_idx], page_idx)

    src.save(str(save_path))
    src.close()
    if same_path:
        save_path.replace(out_path)


def _build_searchable_pdf(
    pdf_path: Path,
    doc,
    out_path: Path,
    raw_cells: dict[int, list] | None = None,
    fallback_pages: set[int] | None = None,
    stamp_page_labels: bool = True,
) -> None:
    """Overlay OCR text as an invisible layer on each page of the PDF.

    Bboxes come from docling's item provenance (paragraph/table/heading
    level) for ordinary pages. Both TOPLEFT and BOTTOMLEFT coordinate origins
    are handled.

    stamp_page_labels=False skips the visible `=== Page N ===` stamp — used
    when this is one chunk of a larger document (see run_ocr's docstring);
    the merge step stamps the correct global page index afterward instead.

    fallback_pages (optional): 0-indexed page numbers where
    _build_page_results decided docling's assembled items dropped text Surya
    actually read (typically a whole page misclassified as a Picture) and
    used raw_cells instead. Without this, those pages got ZERO invisible
    text here even though _build_page_results recovered their text for
    text_docling.txt — this function only ever read doc.iterate_items(),
    which is exactly what's empty/short on those pages. Same fallback here
    keeps the two outputs consistent: a page's searchable-PDF text now always
    matches what text_docling.txt says is on that page.
    """
    import fitz
    from docling_core.types.doc import CoordOrigin

    src = fitz.open(str(pdf_path))
    raw_cells = raw_cells or {}
    fallback_pages = fallback_pages or set()

    # Collect (text, bbox, page_idx) from all document items
    page_items: dict[int, list] = defaultdict(list)
    for item, _ in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        try:
            bbox = prov[0].bbox
            page_no = prov[0].page_no - 1  # 0-indexed
            if bbox is not None:
                page_items[page_no].append((text, bbox))
        except Exception:
            continue

    for page_idx in range(len(src)):
        fitz_page = src[page_idx]
        ph = fitz_page.rect.height  # page height in PDF points (for BOTTOMLEFT conversion)

        if stamp_page_labels:
            _stamp_page_label(fitz_page, page_idx)

        if page_idx in fallback_pages:
            for cell in raw_cells.get(page_idx + 1, []):  # raw_cells is keyed 1-indexed
                text = cell.text
                if not text or not text.strip():
                    continue
                try:
                    bbox = cell.rect.to_bounding_box()  # already TOPLEFT, page points
                    x0, y0, x1, y1 = bbox.l, bbox.t, bbox.r, bbox.b
                    if x1 <= x0 or y1 <= y0 or (y1 - y0) < 2:
                        continue
                    fontsize = max(4.0, (y1 - y0) * 0.8)
                    fitz_page.insert_text(
                        fitz.Point(x0, y1),
                        text,
                        fontsize=fontsize,
                        render_mode=3,   # invisible text (Tr=3)
                        color=(0, 0, 0),
                    )
                except Exception:
                    continue
            continue

        for text, bbox in page_items.get(page_idx, []):
            try:
                origin = getattr(bbox, "coord_origin", CoordOrigin.TOPLEFT)
                if origin == CoordOrigin.BOTTOMLEFT:
                    # PDF native: y increases upward; convert to PyMuPDF top-left
                    x0, y0 = bbox.l, ph - bbox.t
                    x1, y1 = bbox.r, ph - bbox.b
                else:
                    x0, y0, x1, y1 = bbox.l, bbox.t, bbox.r, bbox.b

                if x1 <= x0 or y1 <= y0 or (y1 - y0) < 2:
                    continue

                fontsize = max(4.0, (y1 - y0) * 0.8)
                # insert_text point is the text baseline — approximately the bottom of the bbox
                fitz_page.insert_text(
                    fitz.Point(x0, y1),
                    text,
                    fontsize=fontsize,
                    render_mode=3,   # invisible text (Tr=3)
                    color=(0, 0, 0),
                )
            except Exception:
                continue

    src.save(str(out_path))
    src.close()


def extract_headings_from_txt(txt_path: Path) -> list[dict]:
    """Extract heading lines from text_docling.txt into a flat list.

    Returns [{"page": int, "text": str}, ...] — one entry per heading line,
    in document order. Noise headings are filtered out:
      - purely numeric (e.g. "## 1", "## 4" from OCR'd TOC numbers)
      - very short after stripping (< 4 chars — single letters, punctuation)
      - look like page numbers or math fragments (mostly digits + spaces/punctuation)
    """
    import re
    heading_re = re.compile(r'^#{1,6}\s+(.+)$')
    page_re    = re.compile(r'^===\s*Page\s+(\d+)\s*===$')
    noise_re   = re.compile(r'^[\d\s\.\-\,\:\;\(\)\[\]\{\}\/\\]+$')

    current_page = 0
    headings: list[dict] = []

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        pm = page_re.match(line.strip())
        if pm:
            current_page = int(pm.group(1))
            continue
        hm = heading_re.match(line.strip())
        if hm:
            text = hm.group(1).strip()
            if len(text) < 4:
                continue
            if noise_re.match(text):
                continue
            headings.append({"page": current_page, "text": text})

    return headings


def write_headings_json(txt_path: Path, out_dir: Path, document_name: str) -> Path:
    """Extract headings from txt_path and write headings.json to out_dir."""
    import json
    headings = extract_headings_from_txt(txt_path)
    out = {
        "document": document_name,
        "n_headings": len(headings),
        "headings": headings,
    }
    out_path = out_dir / "headings.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _build_page_results(doc, raw_cells: dict[int, list] | None = None) -> tuple[list[dict], int, set[int]]:
    """Convert a DoclingDocument to per-page result dicts.

    Heading items (TitleItem, SectionHeaderItem) are prefixed with the same
    markdown # markers that docling's export_to_markdown() produces, so the
    text_docling.txt output carries both page boundaries and heading structure
    without requiring cross-referencing against the .md file.

    raw_cells (optional): {page_no (1-indexed): [TextCell, ...]} straight from
    Surya. Docling's layout model sometimes classifies an entire page of dense
    handwriting as a single Picture region, which silently drops every OCR'd
    text cell on that page from doc.iterate_items() even though Surya read it
    correctly. When that happens (the assembled text is shorter than what raw
    OCR actually captured), we fall back to the raw cells for that page so no
    recognized text is lost. Returns (results, n_pages_recovered_via_fallback,
    fallback_page_indices) — the last so _build_searchable_pdf (and the
    per-page Markdown assembly below) can make the identical per-page choice
    instead of re-deriving it independently.

    Each result also carries md_text: real per-page Markdown from
    doc.export_to_markdown(page_no=...) — table structure included — for
    assembled pages, or the raw fallback text wrapped in a fenced code block
    for fallback pages (so a reader can see at a glance which pages got real
    structure recovery vs. a raw OCR dump). This is what <stem>.md is built
    from, so it has the same fallback coverage as text_docling.txt instead
    of silently losing the same ~80%-of-pages content docling's assembled
    path alone loses on handwriting-heavy corpora.
    """
    from docling_core.types.doc import SectionHeaderItem, TitleItem
    from docling_core.types.doc.document import TableItem

    page_heights = _page_heights(doc)
    page_texts: dict[int, list[tuple[str, object]]] = defaultdict(list)
    page_tables: dict[int, list[dict]] = defaultdict(list)

    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        page_no = (prov[0].page_no - 1) if prov else 0  # 0-indexed

        # Tables carry no `.text`, so they must be handled before the text
        # guard below or they are dropped along with their geometry.
        if isinstance(item, TableItem):
            page_tables[page_no].append(_table_to_dict(item, page_heights))
            continue

        text = getattr(item, "text", None)
        if not text:
            continue
        regions = _prov_bboxes(prov, page_heights)

        if isinstance(item, TitleItem):
            text = f"# {text}"
        elif isinstance(item, SectionHeaderItem):
            # Matches docling's own markdown export: level 1 → ##, level 2 → ###, etc.
            num_hashes = min(item.level + 1, 6)
            text = f"{'#' * num_hashes} {text}"

        page_texts[page_no].append((text, regions))

    raw_cells = raw_cells or {}
    max_raw_page_index = max((pn - 1 for pn in raw_cells), default=-1)
    if not page_texts and max_raw_page_index < 0:
        return [], 0, set()

    n_pages = max(max(page_texts.keys(), default=-1), max_raw_page_index) + 1

    results = []
    n_recovered = 0
    fallback_page_indices: set[int] = set()
    for i in range(n_pages):
        assembled_items = page_texts.get(i, [])
        assembled_text = "\n".join(t for t, _ in assembled_items)

        cells = raw_cells.get(i + 1)  # raw_cells is keyed 1-indexed
        if cells:
            fallback_text = _cells_reading_order_text(cells)
            if len(fallback_text.strip()) > len(assembled_text.strip()):
                n_recovered += 1
                fallback_page_indices.add(i)
                results.append({
                    "page_index": i,
                    "image_path": "",
                    "text_lines": [
                        {"text": c.text, "confidence": c.confidence,
                         "bbox": bb, "bboxes": [{**bb, "page": i}] if bb else []}
                        for c, bb in (
                            (c, _bbox_to_dict(c.rect.to_bounding_box(), page_heights.get(i)))
                            for c in sorted(cells, key=lambda c: (round(c.rect.to_bounding_box().t / 15.0), c.rect.to_bounding_box().l))
                            if c.text and c.text.strip()
                        )
                    ],
                    "tables": page_tables.get(i, []),
                    "full_text": fallback_text,
                    "md_text": (
                        "*Raw OCR text — Docling's layout model did not recover "
                        "this page's structure.*\n\n```\n" + fallback_text + "\n```"
                    ),
                })
                continue

        results.append({
            "page_index": i,
            "image_path": "",
            "md_text": doc.export_to_markdown(page_no=i + 1),
            "text_lines": [
                {
                    "text": t, "confidence": 1.0,
                    # `bbox` stays the item's first region, unchanged, so every
                    # existing consumer keeps working; `bboxes` is the whole truth.
                    "bbox": {k: v for k, v in regions[0].items() if k != "page"} if regions else {},
                    "bboxes": regions,
                }
                for t, regions in assembled_items
            ],
            "tables": page_tables.get(i, []),
            "full_text": assembled_text,
        })

    return results, n_recovered, fallback_page_indices
