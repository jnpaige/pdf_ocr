"""Build a searchable PDF from rendered page images + Surya OCR results.

Each page = image background + invisible text layer so the PDF is visually
identical to the scan but fully text-searchable/selectable.

Alignment approach (mirrors what ocrmypdf uses):
  - fontsize  = bbox height           → vertical fit
  - Tz operator                       → horizontal scaling to exact bbox width
  - Tm operator (absolute per line)   → no cumulative position drift
  - render mode 3                     → invisible (painted only by search/select)

Content is written via fitz.TOOLS._insert_contents + doc.update_stream,
the same internal path used by PyMuPDF's Shape.commit().
"""
from pathlib import Path

import fitz
from PIL import Image


def _escape_pdf(text: str) -> str:
    """Escape text for a PDF literal string. Chars outside Latin-1 → space."""
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "(":
            out.append("\\(")
        elif ch == ")":
            out.append("\\)")
        elif ord(ch) > 255:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def build_searchable_pdf(
    image_paths: list[Path],
    ocr_results: list[dict],
    out_path: Path,
    dpi: int,
) -> Path:
    """Write a searchable PDF to *out_path* and return the path."""
    scale = 72.0 / dpi       # image pixels → PDF points
    font = fitz.Font("helv") # used only for text-length measurement
    doc = fitz.open()

    for img_path, result in zip(image_paths, ocr_results):
        # Page dimensions in PDF points.
        # Use PIL for pixel dimensions — fitz.open() on a PNG already converts
        # to points assuming 96 DPI, which would double-scale if multiplied again.
        with Image.open(str(img_path)) as im:
            w_px, h_px = im.size
        w_pts = w_px * scale
        h_pts = h_px * scale

        page = doc.new_page(width=w_pts, height=h_pts)
        page.insert_image(page.rect, filename=str(img_path))

        # Register Helvetica in the page resource dictionary and get its name
        xref = page.insert_font(fontname="helv")
        res_name = next((f[4] for f in page.get_fonts() if f[0] == xref), "helv")

        # Build raw PDF content for the invisible text layer.
        # All lines go in a single BT...ET block to minimise overhead.
        lines = ["q", "BT", "3 Tr"]  # Tr 3 = invisible text

        for line in result["text_lines"]:
            text = line["text"].strip()
            if not text:
                continue

            b = line["bbox"]  # [x_min, y_min, x_max, y_max] in image pixels
            x0    = b[0] * scale
            y1    = b[3] * scale           # bottom of bbox in image-space
            box_w = max((b[2] - b[0]) * scale, 1.0)
            box_h = max((b[3] - b[1]) * scale, 1.0)

            fontsize = box_h

            # Horizontal scale so rendered text width == bbox width exactly
            tw = font.text_length(text, fontsize=fontsize)
            hz = (box_w / tw * 100.0) if tw > 0 else 100.0
            hz = min(max(hz, 1.0), 10000.0)

            # PDF origin is bottom-left; image origin is top-left
            pdf_y = h_pts - y1

            # Tf sets font + size.
            # Tm uses identity matrix (1 0 0 1) for rotation/scale so fontsize
            # is only applied once — putting fontsize in both Tf and Tm would
            # square the effective size.
            # Tz stretches/compresses glyphs horizontally to fill bbox width.
            lines += [
                f"/{res_name} {fontsize:.3f} Tf",
                f"{hz:.3f} Tz",
                f"1 0 0 1 {x0:.3f} {pdf_y:.3f} Tm",
                f"({_escape_pdf(text)}) Tj",
            ]

        lines += ["ET", "Q"]
        content = "\n".join(lines).encode()

        # Append our content stream to the page (same path as Shape.commit)
        page.wrap_contents()
        stream_xref = fitz.TOOLS._insert_contents(page, b" ", True)
        doc.update_stream(stream_xref, content)

    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return out_path
