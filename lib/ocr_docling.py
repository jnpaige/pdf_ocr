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
def run_ocr(pdf_path: Path, out_dir: Path, docling_cfg: dict | None = None, stem: str | None = None) -> list[dict]:
    """Run Docling + Surya on a PDF. Saves <stem>.md to out_dir.

    stem overrides pdf_path.stem for output file naming — used in backfill mode
    where the input is a *_ocr.pdf but outputs should use the canonical report name.

    Returns per-page result dicts with keys:
      page_index, image_path, text_lines, full_text
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions

    if docling_cfg is None:
        docling_cfg = {}
    if stem is None:
        stem = pdf_path.stem

    do_ocr = docling_cfg.get("do_ocr", True)
    extract_figures = docling_cfg.get("extract_figures", False)
    picture_description_model = docling_cfg.get("picture_description_model")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr

    if extract_figures:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = docling_cfg.get("figures_scale", 2.0)
        pipeline_options.do_picture_classification = docling_cfg.get("classify_figures", False)

    if picture_description_model:
        pipeline_options.enable_remote_services = True
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = PictureDescriptionApiOptions(
            url=docling_cfg.get("picture_description_base_url", "http://localhost:11434") + "/v1/chat/completions",
            params={"model": picture_description_model},
            prompt=docling_cfg.get(
                "picture_description_prompt",
                "Describe this figure from an archaeology report in 1-3 sentences. "
                "If it shows a lithic artifact, note the artifact type, reduction "
                "technique, or technological features visible (e.g. platform, "
                "bulb of percussion, retouch, cortex). If it is a map, chart, or "
                "photo of a site or excavation, say so plainly instead.",
            ),
            timeout=docling_cfg.get("picture_description_timeout", 120),
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=SuryaPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    _raw_ocr_cells_by_page.clear()

    if do_ocr:
        print(f"  Running Docling + Surya OCR...")
    else:
        print(f"  Running Docling (using existing text layer, do_ocr: false)...")
    result = converter.convert(pdf_path)
    doc = result.document

    md_content = doc.export_to_markdown()
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

    if do_ocr:
        pdf_out = out_dir / f"{stem}_ocr.pdf"
        _build_searchable_pdf(pdf_path, doc, pdf_out)
        print(f"  Saved searchable PDF → {pdf_out.name}")
    else:
        print(f"  Skipping searchable PDF (do_ocr: false — input already has a text layer)")

    results, n_recovered = _build_page_results(doc, raw_cells=_raw_ocr_cells_by_page)
    if n_recovered:
        print(f"  Recovered {n_recovered} page(s) via raw OCR fallback "
              f"(layout model produced little/no text there)")

    # Write headings.json alongside the other outputs
    txt_path = out_dir / "text_docling.txt"
    if txt_path.exists():
        h_path = write_headings_json(txt_path, out_dir, pdf_path.stem)
        print(f"  Saved headings    → {h_path.name}  ({_count_headings(h_path)} headings)")

    if extract_figures:
        n_figs = _extract_figures(doc, out_dir, captioned=bool(picture_description_model))
        print(f"  Saved figures     → figures/  ({n_figs} figure(s))")

    return results


def _extract_figures(doc, out_dir: Path, captioned: bool = False) -> int:
    """Save each detected picture/figure as a PNG under <out_dir>/figures/, plus
    a figures.json manifest recording page, bbox, classification (if enabled),
    and VLM caption (if picture_description_model was set). Returns the count
    of figures saved.
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

        if captioned:
            description = getattr(pic.meta, "description", None) if pic.meta else None
            entry["caption"] = description.text if description else None

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


def _build_searchable_pdf(pdf_path: Path, doc, out_path: Path) -> None:
    """Overlay Surya's OCR text as an invisible layer on each page of the PDF.

    Bboxes come from docling's item provenance (paragraph/table/heading level).
    Both TOPLEFT and BOTTOMLEFT coordinate origins are handled.
    """
    import fitz
    from docling_core.types.doc import CoordOrigin

    src = fitz.open(str(pdf_path))

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


def _build_page_results(doc, raw_cells: dict[int, list] | None = None) -> tuple[list[dict], int]:
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
    recognized text is lost. Returns (results, n_pages_recovered_via_fallback).
    """
    from docling_core.types.doc import SectionHeaderItem, TitleItem

    page_texts: dict[int, list[str]] = defaultdict(list)

    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None)
        page_no = (prov[0].page_no - 1) if prov else 0  # 0-indexed

        if isinstance(item, TitleItem):
            text = f"# {text}"
        elif isinstance(item, SectionHeaderItem):
            # Matches docling's own markdown export: level 1 → ##, level 2 → ###, etc.
            num_hashes = min(item.level + 1, 6)
            text = f"{'#' * num_hashes} {text}"

        page_texts[page_no].append(text)

    raw_cells = raw_cells or {}
    max_raw_page_index = max((pn - 1 for pn in raw_cells), default=-1)
    if not page_texts and max_raw_page_index < 0:
        return [], 0

    n_pages = max(max(page_texts.keys(), default=-1), max_raw_page_index) + 1

    results = []
    n_recovered = 0
    for i in range(n_pages):
        assembled_lines = page_texts.get(i, [])
        assembled_text = "\n".join(assembled_lines)

        cells = raw_cells.get(i + 1)  # raw_cells is keyed 1-indexed
        if cells:
            fallback_text = _cells_reading_order_text(cells)
            if len(fallback_text.strip()) > len(assembled_text.strip()):
                n_recovered += 1
                results.append({
                    "page_index": i,
                    "image_path": "",
                    "text_lines": [
                        {"text": c.text, "confidence": c.confidence, "bbox": []}
                        for c in sorted(cells, key=lambda c: (round(c.rect.to_bounding_box().t / 15.0), c.rect.to_bounding_box().l))
                        if c.text and c.text.strip()
                    ],
                    "full_text": fallback_text,
                })
                continue

        results.append({
            "page_index": i,
            "image_path": "",
            "text_lines": [{"text": t, "confidence": 1.0, "bbox": []} for t in assembled_lines],
            "full_text": assembled_text,
        })

    return results, n_recovered
