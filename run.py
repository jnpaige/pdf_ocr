"""
run.py — PDF to OCR pipeline
Renders a PDF to page images at a specified DPI, then OCRs each page with
one or more engines (surya, kraken) as configured in config.yaml.

Usage:
    python run.py
    python run.py --config path/to/config.yaml

Output folder: <output_dir>/<pdf_stem>_<dpi>dpi_<engines>/
  page_000.png, page_001.png, ...       rendered page images
  ocr_<engine>.json                     structured results per engine
  text_<engine>.txt                     plain text per engine
  <pdf_stem>_ocr_<engine>.pdf           searchable PDF per engine

Edit config.yaml to change the input PDF, DPI, output location, or engines.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from render import render_pdf
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


def output_folder_name(pdf_path: Path, dpi: int, engines: list[str]) -> str:
    return f"{pdf_path.stem}_{dpi}dpi_{'_'.join(engines)}"


def save_engine_outputs(
    out_dir: Path,
    engine: str,
    ocr_results: list[dict],
    image_paths: list[Path],
    dpi: int,
    pdf_stem: str,
):
    ocr_path = out_dir / f"ocr_{engine}.json"
    ocr_path.write_text(json.dumps(ocr_results, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = out_dir / f"text_{engine}.txt"
    sections = [f"=== Page {r['page_index']} ===\n{r['full_text']}" for r in ocr_results]
    text_path.write_text("\n\n".join(sections), encoding="utf-8")

    pdf_out = build_searchable_pdf(
        image_paths, ocr_results, out_dir / f"{pdf_stem}_ocr_{engine}.pdf", dpi
    )

    print(f"    Saved: {ocr_path.name}, {text_path.name}, {pdf_out.name}")


def process_pdf(pdf_path: Path, cfg: dict):
    dpi        = int(cfg.get("page_image_dpi", 150))
    output_dir = Path(cfg["output_dir"])
    engines    = cfg.get("ocr_engines", ["surya"])

    out_dir = output_dir / output_folder_name(pdf_path, dpi, engines)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{pdf_path.name}]")
    print(f"  Rendering at {dpi} DPI → {out_dir}")
    image_paths = render_pdf(pdf_path, out_dir, dpi)
    print(f"  Rendered {len(image_paths)} page(s)")

    for engine in engines:
        print(f"\n  [{engine.upper()}]")

        if engine == "surya":
            from ocr_surya import run_ocr as surya_ocr
            languages  = cfg.get("ocr_languages", ["en"])
            surya_cfg  = cfg.get("surya", {})
            results    = surya_ocr(image_paths, languages, surya_cfg=surya_cfg)

        elif engine == "mineru":
            from ocr_mineru import run_ocr as mineru_ocr
            languages   = cfg.get("ocr_languages", ["en"])
            mineru_cfg  = cfg.get("mineru", {})
            mineru_work = out_dir / "mineru_work"
            results     = mineru_ocr(pdf_path, dpi, mineru_work, languages,
                                     mineru_cfg=mineru_cfg)
            # Stitch rendered image paths into results (MinerU doesn't know about them)
            for i, r in enumerate(results):
                if i < len(image_paths):
                    r["image_path"] = str(image_paths[i])

        else:
            print(f"  Unknown engine '{engine}', skipping.")
            continue

        save_engine_outputs(out_dir, engine, results, image_paths, dpi, pdf_path.stem)

    print(f"\n  Done → {out_dir}")


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