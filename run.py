"""
run.py — PDF to OCR pipeline
Renders a PDF to page images at a specified DPI, then OCRs each page with Surya.

Usage:
    python run.py
    python run.py --config path/to/config.yaml

Output folder: <output_dir>/<pdf_stem>_<dpi>dpi_surya/
  page_000.png, page_001.png, ...   rendered page images
  ocr.json                          structured OCR results (text, bboxes, confidence)
  text.txt                          plain concatenated text, one page per section

Edit config.yaml to change the input PDF, DPI, or output location.
"""


import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from render import render_pdf
from ocr_surya import run_ocr
from pdf_writer import build_searchable_pdf


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_pdfs(pdf_input: str) -> list[Path]:
    p = Path(pdf_input)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.pdf"))
    raise ValueError(f"pdf_input must be a .pdf file or a directory: {pdf_input}")


def output_folder_name(pdf_path: Path, dpi: int, ocr_method: str = "surya") -> str:
    return f"{pdf_path.stem}_{dpi}dpi_{ocr_method}"


def save_outputs(out_dir: Path, ocr_results: list[dict], image_paths: list[Path], dpi: int, pdf_stem: str):
    ocr_path = out_dir / "ocr.json"
    ocr_path.write_text(json.dumps(ocr_results, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = out_dir / "text.txt"
    sections = []
    for r in ocr_results:
        sections.append(f"=== Page {r['page_index']} ===\n{r['full_text']}")
    text_path.write_text("\n\n".join(sections), encoding="utf-8")

    pdf_path = build_searchable_pdf(image_paths, ocr_results, out_dir / f"{pdf_stem}_ocr.pdf", dpi)

    print(f"  Saved: {ocr_path.name}, {text_path.name}, {pdf_path.name}")


def process_pdf(pdf_path: Path, cfg: dict):
    dpi        = int(cfg.get("page_image_dpi", 70))
    output_dir = Path(cfg["output_dir"])
    languages  = cfg.get("ocr_languages", ["en"])

    out_dir = output_dir / output_folder_name(pdf_path, dpi)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{pdf_path.name}]")
    print(f"  Rendering at {dpi} DPI → {out_dir}")
    image_paths = render_pdf(pdf_path, out_dir, dpi)
    print(f"  Rendered {len(image_paths)} page(s)")

    ocr_results = run_ocr(image_paths, languages)
    save_outputs(out_dir, ocr_results, image_paths, dpi, pdf_path.stem)  # pdf_path.stem = original PDF filename
    print(f"  Done → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: config.yaml alongside run.py)",
    )
    args = parser.parse_args()

    cfg  = load_config(Path(args.config))
    pdfs = collect_pdfs(cfg["pdf_input"])

    print(f"Found {len(pdfs)} PDF(s)")
    for pdf_path in pdfs:
        process_pdf(pdf_path, cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()
