"""Run MinerU (magic-pdf) OCR on a PDF and return structured results.

MinerU uses PaddleOCR for text recognition and its own layout model for
multi-column segmentation, figure/table separation, and reading-order recovery.
It operates on the PDF directly (pre-rendered images are not required for OCR).

Installation
------------
    pip install "magic-pdf[full]"
    # Then download model weights — see:
    # https://github.com/opendatalab/MinerU#quick-start

Coordinate system
-----------------
MinerU stores bboxes in PDF point units (72 pt = 1 inch) with top-left origin.
This module translates them to pixel coordinates at the caller-specified DPI so
they align with the rendered page images used by build_searchable_pdf.
"""
import json
import shutil
import subprocess
from pathlib import Path


_INSTALL_MSG = (
    "magic-pdf CLI not found.\n"
    "Install MinerU:   pip install 'magic-pdf[full]'\n"
    "Download models:  https://github.com/opendatalab/MinerU#quick-start"
)


def run_ocr(
    pdf_path: Path,
    dpi: int,
    work_dir: Path,
    languages: list[str],
    mineru_cfg: dict | None = None,
) -> list[dict]:
    """OCR a PDF with MinerU and return one result dict per page.

    Parameters
    ----------
    pdf_path : Path
        Input PDF file.
    dpi : int
        DPI used to render page images.  MinerU bboxes (in PDF points) are
        scaled by dpi/72 to match the pixel coordinate space of those images.
    work_dir : Path
        Directory where MinerU writes intermediate outputs.  Preserved across
        runs so the CLI call is skipped if middle.json already exists.
    languages : list[str]
        Informational only — MinerU language selection is configured via its
        own config file, not a CLI flag.
    mineru_cfg : dict, optional
        The 'mineru' block from config.yaml.  Recognised keys:
          method : "ocr" | "auto" | "txt"
                   "ocr"  — force PaddleOCR (correct for scanned PDFs)
                   "auto" — MinerU decides based on PDF content
                   "txt"  — extract embedded text only, no OCR
    """
    if mineru_cfg is None:
        mineru_cfg = {}

    if shutil.which("magic-pdf") is None:
        raise RuntimeError(_INSTALL_MSG)

    method = mineru_cfg.get("method", "ocr")
    work_dir.mkdir(parents=True, exist_ok=True)

    # MinerU writes to {work_dir}/{pdf_stem}/{method}/
    stem = pdf_path.stem
    expected = work_dir / stem / method / f"{stem}_middle.json"

    if not expected.exists():
        print(f"  Running MinerU (method={method}) on {pdf_path.name}...")
        cmd = ["magic-pdf", "-p", str(pdf_path), "-o", str(work_dir), "-m", method]
        # Do NOT capture output — let magic-pdf write directly to the terminal so
        # any crash traceback is fully visible rather than silently swallowed.
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            raise RuntimeError(
                f"magic-pdf exited with code {proc.returncode} — see output above."
            )
    else:
        print(f"  MinerU output found, skipping CLI call.")

    # Glob for middle.json — MinerU may sanitize the PDF stem (spaces → underscores,
    # etc.) so the filename won't always match our expected path exactly.
    middle_path = expected if expected.exists() else next(
        iter(work_dir.rglob("*_middle.json")), None
    )
    if middle_path is None:
        # Show what was actually written to help diagnose the problem
        all_files = list(work_dir.rglob("*"))
        file_list = "\n".join(f"    {f.relative_to(work_dir)}" for f in all_files) \
                    or "    (nothing written)"
        raise RuntimeError(
            f"No *_middle.json found under {work_dir} after MinerU ran.\n"
            f"Files present in work_dir:\n{file_list}\n"
            "Possible causes: models not downloaded, or magic-pdf install incomplete.\n"
            "Run:  magic-pdf --help   to check for first-run model download prompts."
        )

    print(f"  Parsing MinerU output: {middle_path.name}")
    with open(middle_path, "r", encoding="utf-8") as f:
        middle_data = json.load(f)

    return _parse_middle(middle_data, dpi)


def _parse_middle(middle_data: dict, dpi: int) -> list[dict]:
    """Convert MinerU middle.json content to the standard result-dict list."""
    scale = dpi / 72.0  # convert PDF points → pixels at target DPI

    results = []
    for page in middle_data.get("pdf_info", []):
        page_idx = page.get("page_no", len(results))
        page_lines = []

        for block in page.get("para_blocks", []):
            block_type = block.get("type", "text")

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_text = "".join(s.get("content", "") for s in spans)
                if not line_text.strip():
                    continue

                lb = line.get("bbox") or block.get("bbox")
                if lb is None or len(lb) < 4:
                    continue

                scores = [s["score"] for s in spans if "score" in s]
                confidence = sum(scores) / len(scores) if scores else 1.0

                page_lines.append({
                    "text":       line_text,
                    "confidence": round(confidence, 4),
                    "bbox":       [round(lb[0] * scale), round(lb[1] * scale),
                                   round(lb[2] * scale), round(lb[3] * scale)],
                    "region":     block_type,
                })

        results.append({
            "page_index": page_idx,
            "image_path": "",   # stitched in by run.py after render_pdf
            "text_lines": page_lines,
            "full_text":  "\n".join(l["text"] for l in page_lines),
        })

    return results