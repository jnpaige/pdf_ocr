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
def run_ocr(pdf_path: Path, out_dir: Path, docling_cfg: dict | None = None) -> list[dict]:
    """Run Docling + Surya on a PDF. Saves <stem>.md to out_dir.

    Returns per-page result dicts with keys:
      page_index, image_path, text_lines, full_text
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    if docling_cfg is None:
        docling_cfg = {}

    do_ocr = docling_cfg.get("do_ocr", True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=SuryaPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    if do_ocr:
        print(f"  Running Docling + Surya OCR...")
    else:
        print(f"  Running Docling (using existing text layer, do_ocr: false)...")
    result = converter.convert(pdf_path)
    doc = result.document

    md_content = doc.export_to_markdown()
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Saved markdown → {md_path.name}")

    flat_md_dir = docling_cfg.get("markdown_dir")
    if flat_md_dir:
        flat_dir = Path(flat_md_dir)
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_path = flat_dir / f"{pdf_path.stem}.md"
        flat_path.write_text(md_content, encoding="utf-8")
        print(f"  Mirrored markdown → {flat_path}")

    if do_ocr:
        pdf_out = out_dir / f"{pdf_path.stem}_ocr.pdf"
        _build_searchable_pdf(pdf_path, doc, pdf_out)
        print(f"  Saved searchable PDF → {pdf_out.name}")
    else:
        print(f"  Skipping searchable PDF (do_ocr: false — input already has a text layer)")

    results = _build_page_results(doc)

    # Write headings.json alongside the other outputs
    txt_path = out_dir / "text_docling.txt"
    if txt_path.exists():
        h_path = write_headings_json(txt_path, out_dir, pdf_path.stem)
        print(f"  Saved headings    → {h_path.name}  ({_count_headings(h_path)} headings)")

    return results


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


def _build_page_results(doc) -> list[dict]:
    """Convert a DoclingDocument to per-page result dicts.

    Heading items (TitleItem, SectionHeaderItem) are prefixed with the same
    markdown # markers that docling's export_to_markdown() produces, so the
    text_docling.txt output carries both page boundaries and heading structure
    without requiring cross-referencing against the .md file.
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

    if not page_texts:
        return []

    n_pages = max(page_texts.keys()) + 1
    return [
        {
            "page_index": i,
            "image_path": "",
            "text_lines": [{"text": t, "confidence": 1.0, "bbox": []} for t in page_texts.get(i, [])],
            "full_text":  "\n".join(page_texts.get(i, [])),
        }
        for i in range(n_pages)
    ]
