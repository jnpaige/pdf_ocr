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


def _safe_convert_path(pdf_path: Path) -> tuple[Path, Path | None]:
    """Return a path safe to hand to Docling's PDF backend, plus a temp file
    to delete afterward (None if pdf_path was already safe, i.e. the common
    case — no copy made).

    Docling's PDFium-based backend can fail to open a PDF whose filename
    contains certain non-ASCII characters — observed on Windows with a
    Unicode right single quote (U+2019) in an author name, e.g.
    "...Ifri n'Etsedda.pdf" — raising `ConversionError: ... is not valid`,
    even though PyMuPDF opens the identical bytes without complaint and
    reports the correct page count. This is a path-handling limitation of
    the backend, not a property of the PDF content, but it's fatal to the
    whole run once it happens: run.py's main loop does not catch per-PDF
    exceptions, so one such file kills every PDF still queued behind it.

    Academic literature filenames routinely carry diacritics and special
    characters in author names, so this isn't a rare edge case for this
    pipeline's actual corpora — copying to a plain-ASCII temp filename
    before conversion (and converting from that copy) sidesteps the bug
    entirely rather than dropping every affected document.
    """
    try:
        str(pdf_path).encode("ascii")
        return pdf_path, None
    except UnicodeEncodeError:
        pass

    import shutil
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
    import os
    os.close(fd)
    tmp_path = Path(tmp_name)
    shutil.copy(pdf_path, tmp_path)
    return tmp_path, tmp_path


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

    convert_path, tmp_cleanup = _safe_convert_path(pdf_path)
    try:
        result = converter.convert(convert_path)
    finally:
        if tmp_cleanup is not None:
            tmp_cleanup.unlink(missing_ok=True)
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
        delete_classes = set(docling_cfg.get("delete_figure_classes", []))
        n_figs = _extract_figures(doc, out_dir, delete_figure_classes=delete_classes)
        print(f"  Saved figures     → figures/  ({n_figs} figure(s))")

    return results


def _snippet(text: str | None, max_chars: int = 500) -> str | None:
    if not text:
        return None
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _figure_context(doc) -> dict[str, dict]:
    """Walk the document once in reading order to recover, for every picture,
    the context a human (or a VLM downstream) would use to interpret it: the
    active heading path, and the body text immediately before/after it.

    PictureItem itself carries none of this — only its own bbox/page — so
    this has to be reconstructed from position in doc.iterate_items() rather
    than read off the item directly. heading_path[i] is the heading active at
    document level i (index 0 = document title, if any); it's a best-effort
    reconstruction that assumes headings are properly nested (docling levels
    don't skip), same assumption the rest of this file already makes about
    heading structure (see extract_headings_from_txt).

    Also resolves each picture's caption here via pic.caption_text(doc)
    directly. An earlier version tried to recover captions Docling didn't
    associate (e.g. a multi-panel figure detected as several independent
    PictureItems sharing one printed caption, attached to only one of them)
    by inheriting from reading-order neighbors. Checked against a real
    corpus: it recovered a real but modest fraction of the gap (6 of 147
    figures) and never touched documents whose uncaptioned figures don't
    happen to sit next to a captioned sibling (most of the missing ones, in
    practice) — not enough yield to justify the extra mechanism, especially
    now that a missing caption is just an honestly-empty box on the saved
    image rather than a silently-absent field. Reverted in favor of trusting
    Docling's own association plainly; ~30% of figures across a real corpus
    end up with caption: None, and that's left as-is.

    Returns {picture.self_ref: {"heading_path": [...], "text_before": str|None,
    "text_after": str|None, "caption": str|None}}.
    """
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TitleItem

    # A figure's own caption is almost always the very next text item in
    # reading order, which made text_after just re-derive the caption field
    # verbatim. Caption items are already captured separately (caption_text),
    # so they're excluded here — text_before/text_after should only ever be
    # genuine surrounding body text, not the figure's own caption restated.
    caption_refs: set[str] = {
        cap.cref for pic in doc.pictures for cap in pic.captions
    }

    context: dict[str, dict] = {}
    heading_stack: list[str] = []
    prev_text: str | None = None
    pending_refs: list[str] = []  # pictures awaiting the next text item for text_after

    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem):
            context[item.self_ref] = {
                "heading_path": list(heading_stack),
                "text_before": _snippet(prev_text),
                "text_after": None,
                "caption": item.caption_text(doc) or None,
            }
            pending_refs.append(item.self_ref)
            continue

        if item.self_ref in caption_refs:
            continue

        if isinstance(item, TitleItem):
            heading_stack = [item.text]
        elif isinstance(item, SectionHeaderItem):
            heading_stack = heading_stack[: item.level] + [item.text]

        text = getattr(item, "text", None)
        if text and text.strip():
            for ref in pending_refs:
                context[ref]["text_after"] = _snippet(text)
            pending_refs.clear()
            prev_text = text

    return context


# Docling's DocumentFigureClassifier-v2.5 labels that are publisher/platform
# chrome rather than a real figure — verified by hand against a real corpus
# (every "logo"/"icon" sample checked was a journal masthead banner or the
# CrossRef "Check for updates" badge, never an actual figure) before trusting
# this as grounds for deletion rather than just a flag. Only `logo` and
# `icon` are deleted by default; the rest of this set exists so a caller can
# opt into the wider net (docling.delete_figure_classes in config) without
# having to know Docling's full label list.
JUNK_FIGURE_CLASSES = {
    "logo", "icon", "page_thumbnail", "qr_code", "bar_code",
    "stamp", "signature", "screenshot_from_computer",
    "screenshot_from_manual", "calendar", "crossword_puzzle", "music",
}


def prune_figures_by_class(figures_dir: Path, manifest: dict, junk_classes: set[str]) -> tuple[dict, int]:
    """Delete figure PNGs whose Docling classification is in junk_classes and
    drop them from the manifest. Returns (updated_manifest, n_deleted).

    Retroactive cleanup only — a run made before delete_figure_classes
    existed, or one made with a narrower junk set than you want now. New
    runs skip these figures before ever saving them (see
    _extract_figures' delete_figure_classes param) rather than saving then
    deleting; this function exists so an already-extracted corpus doesn't
    have to be re-OCR'd from scratch just to apply that same policy after
    the fact.
    """
    kept: list[dict] = []
    n_deleted = 0
    for entry in manifest.get("figures", []):
        if entry.get("classification") in junk_classes:
            (figures_dir / entry["file"]).unlink(missing_ok=True)
            n_deleted += 1
        else:
            kept.append(entry)
    manifest["figures"] = kept
    manifest["n_figures"] = len(kept)
    return manifest, n_deleted


def _parse_figure_number(caption: str | None) -> int | None:
    """Pull the printed figure number off the front of a caption, e.g.
    "Fig. 3. Artifacts from..." -> 3, "Figure 12: " -> 12. Returns None if
    the caption is missing or doesn't start with a recognizable "Fig[ure]
    N" label (multi-panel figures, uncaptioned plates, non-English papers,
    etc.) — those fall back to positional numbering in _extract_figures.
    """
    import re

    if not caption:
        return None
    m = re.match(r'^fig(?:ure)?\.?\s*(\d+)', caption.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def stamp_figure_label(image, label: str):
    """Stamp `label` (e.g. "Fig 2 - p3") in the top-right corner of a figure
    image — the same troubleshooting idea as _stamp_page_label's
    `=== Page N ===` PDF stamp, applied to figures: a bare image opened out
    of context (e.g. inside a VLM prompt, or a review tool) can still be
    matched back to its assigned figure/page by eye, without needing the
    filename visible. Returns the (possibly mode-converted) image; call
    before image.save() and use the returned image, not the original — this
    does not mutate in place if a mode conversion was needed.
    """
    from PIL import ImageDraw, ImageFont

    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    fontsize = max(10, min(18, image.width // 40))
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    x2, y1 = image.width - 6, 6
    x1, y2 = x2 - tw - 2 * pad, y1 + th + 2 * pad
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    draw.text((x1 + pad, y1 + pad - bbox[1]), label, fill=(0, 0, 0), font=font)
    return image


def compose_figure_with_caption(image, caption: str | None):
    """Append `caption` as a same-width text box beneath a figure image, so
    the two travel as one self-contained file — opening the PNG in any
    viewer shows the caption directly, and a downstream VLM call that only
    receives the image file still has the caption text in front of it.

    Caption text and figure image can otherwise get separated (two
    independent pieces of data, only ever joined via figures.json) or
    simply never looked at together by a human browsing the figures/
    folder. This makes that pairing physically permanent. Returns `image`
    unchanged if there's no caption to add — Docling doesn't always
    associate one (~30% of figures across a real corpus; see
    _figure_context).
    """
    from PIL import ImageDraw, ImageFont
    import textwrap

    if image.mode != "RGB":
        image = image.convert("RGB")
    if not caption:
        return image

    width = image.width
    fontsize = max(14, min(22, width // 45))
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    tmp_draw = ImageDraw.Draw(image)
    avg_char_w = tmp_draw.textlength("x" * 40, font=font) / 40
    wrap_chars = max(20, int(width * 0.94 / avg_char_w))

    wrapped_lines: list[str] = []
    for para in caption.split("\n"):
        wrapped_lines.extend(textwrap.wrap(para, width=wrap_chars) or [""])

    line_height = int(fontsize * 1.4)
    pad = 14
    box_height = pad * 2 + line_height * len(wrapped_lines)

    composite = Image.new("RGB", (width, image.height + box_height), (255, 255, 255))
    composite.paste(image, (0, 0))
    draw = ImageDraw.Draw(composite)
    draw.line([(0, image.height), (width, image.height)], fill=(0, 0, 0), width=2)

    y = image.height + pad
    for line in wrapped_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    return composite


def _save_image_with_retry(image, path: Path, attempts: int = 4, delay: float = 0.5) -> None:
    """image.save() with a short retry-with-backoff, for output directories
    that live under a sync client (Box Drive, OneDrive, ...).

    Seen in practice: two full-corpus runs each failed on exactly one figure
    save with OSError [Errno 22] Invalid argument, on a different filename
    each time, only ever on the Box-synced output path — the identical
    image object saved without error to a local disk path in isolation.
    That signature (non-deterministic file, environment-specific, image
    itself fine) points at the sync client transiently holding the file
    rather than anything wrong with the image, so retrying after a brief
    pause is the right fix, not a code change to how the image is built.
    """
    import time

    last_err: OSError | None = None
    for attempt in range(attempts):
        try:
            image.save(path)
            return
        except OSError as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(delay)
    raise last_err


def _extract_figures(doc, out_dir: Path, delete_figure_classes: set[str] | None = None) -> int:
    """Save each detected picture/figure as a PNG under <out_dir>/figures/, plus
    a figures.json manifest recording page, bbox, classification (if enabled),
    the figure's own printed caption (if Docling found one), its active
    heading path, and a short snippet of body text immediately before/after
    it. Returns the count of figures kept (after any deletion below).

    The caption/heading/context fields describe what's already on the page —
    they make each manifest entry a self-contained unit a downstream VLM
    pass can prompt from without re-reading the whole document. This is
    distinct from a model-GENERATED caption: figure_captioner (or any future
    downstream coding pass) writes that as its own
    <model_slug>__<figure_stem>.caption.json next to these files rather than
    editing this manifest, so nothing here is overwritten by that later step.

    The saved PNG itself also carries the resolved caption text, appended
    as a same-width box beneath the figure (compose_figure_with_caption) —
    opening a figure file directly, e.g. from the figures/ folder, shows
    its caption without needing to cross-reference figures.json. Skipped
    when Docling didn't associate a caption with this picture at all (see
    _figure_context) — left as a plain figure with no box, not a guess.

    delete_figure_classes: if given (requires classify_figures: true to have
    populated the classification field in the first place), any figure whose
    Docling classification is in this set is skipped entirely — never
    rendered, never saved to disk, never in the manifest — rather than saved
    and then deleted. Classification runs as a docling enrichment stage
    *before* this function is called, so every picture already has its
    classification available up front; there's no reason to pay for
    get_image()'s render/crop or write a PNG for something about to be
    thrown away. See prune_figures_by_class/JUNK_FIGURE_CLASSES for the
    equivalent retroactive cleanup of a figures.json from a run that
    predates this setting.

    Filenames are `fig<N>_p<page>.png`, N an assumed sequential number — the
    1st, 2nd, 3rd, ... kept figure in document reading order — not the
    figure's own printed caption number. An earlier version tried to parse
    the real number off the caption (e.g. "Fig. 3. Artifacts..." -> 3) and
    fall back to a separate "unlabeled" counter when that failed, but
    Docling's caption-to-picture association misses often enough in
    practice that most figures ended up in the unlabeled bucket anyway —
    two parallel numbering schemes with the "real" one rarely firing wasn't
    worth the complexity. _parse_figure_number's result is still recorded
    per-figure as `printed_figure_number` (None when not parseable) so a
    caption-derived number remains available for QA/cross-checking without
    driving the filename. Because N is now always assigned (never skipped
    or letter-suffixed), it stays contiguous among survivors automatically
    — deleting a logo/icon before this point just means one fewer picture
    in the loop, not a gap.
    """
    import json

    delete_figure_classes = delete_figure_classes or set()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    context_by_ref = _figure_context(doc)

    manifest: list[dict] = []
    n_skipped = 0
    n = 0  # assumed sequential figure number, 1-indexed, kept figures only
    for pic in doc.pictures:
        classification = getattr(pic.meta, "classification", None) if pic.meta else None
        class_name = classification.get_main_prediction().class_name if classification is not None else None

        if class_name in delete_figure_classes:
            n_skipped += 1
            continue

        image = pic.get_image(doc)
        if image is None:
            continue

        prov = pic.prov[0] if pic.prov else None
        page_no = (prov.page_no - 1) if prov else None  # 0-indexed, matches text_docling.txt
        bbox = prov.bbox if prov else None

        ctx = context_by_ref.get(pic.self_ref, {})
        caption = ctx.get("caption")
        printed_fig_num = _parse_figure_number(caption)

        n += 1
        page_label = page_no if page_no is not None else 'x'
        fname = f"fig{n}_p{page_label}.png"

        image = stamp_figure_label(image, f"Fig {n} - p{page_label}")
        image = compose_figure_with_caption(image, caption)
        _save_image_with_retry(image, figures_dir / fname)

        entry = {
            "file": fname,
            "page": page_no,
            "bbox": [bbox.l, bbox.t, bbox.r, bbox.b] if bbox else None,
        }

        if class_name is not None:
            entry["classification"] = class_name

        entry["caption"] = caption if caption else None
        entry["printed_figure_number"] = printed_fig_num

        entry["heading_path"] = ctx.get("heading_path", [])
        entry["text_before"] = ctx.get("text_before")
        entry["text_after"] = ctx.get("text_after")

        manifest.append(entry)

    if n_skipped:
        print(f"  Skipped {n_skipped} figure(s) classified as {sorted(delete_figure_classes)} (not saved)")

    out = {"n_figures": len(manifest), "figures": manifest}
    (figures_dir / "figures.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out["n_figures"]


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
