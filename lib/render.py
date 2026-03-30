"""Render PDF pages to PNG images at a specified DPI using PyMuPDF."""
from pathlib import Path

import fitz  # pymupdf


def render_pdf(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Render every page of pdf_path to out_dir/page_NNN.png at dpi.

    Skips pages that already exist on disk. Returns the list of image paths
    in page order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(str(pdf_path))
    paths = []

    for i, page in enumerate(doc):
        img_path = out_dir / f"page_{i:03d}.png"
        if not img_path.exists():
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            pix.save(str(img_path))
        paths.append(img_path)

    return paths