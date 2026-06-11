"""
run.py — PDF to Markdown pipeline (Docling + Surya OCR)

Processes a single PDF or a directory of PDFs. Each PDF is passed through the
Docling pipeline with Surya as the OCR backend, producing a Markdown file per
PDF suitable for RAG indexing or LLM agent ingestion.

Usage:
    python run.py
    python run.py --config path/to/config.yaml

Output per PDF:  <output_dir>/<pdf_stem>/
    <pdf_stem>.md        Markdown with layout, tables, and reading order preserved
    ocr_docling.json     Structured per-page results
    text_docling.txt     Plain text (one section per page)
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "lib"))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_pdfs(
    pdf_input: str,
    start_from: str | None = None,
    file_list: str | None = None,
) -> list[Path]:
    p = Path(pdf_input)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if not p.is_dir():
        raise ValueError(f"pdf_input must be a .pdf file or a directory: {pdf_input}")

    if file_list:
        names = [
            line.strip()
            for line in Path(file_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pdfs = []
        for name in names:
            stem = Path(name).stem
            candidate = p / f"{stem}.pdf"
            if candidate.exists():
                pdfs.append(candidate)
            else:
                print(f"  WARNING: listed file not found, skipping: {candidate}")
        return pdfs

    pdfs = sorted(p.glob("*.pdf"), key=lambda x: x.stem.upper())
    if start_from:
        cutoff = start_from.upper()
        pdfs = [f for f in pdfs if f.stem.upper() >= cutoff]
    return pdfs


def is_already_done(pdf_path: Path, cfg: dict) -> bool:
    out_dir = Path(cfg["output_dir"]) / pdf_path.stem
    return (out_dir / "ocr_docling.json").exists()


def process_pdf(pdf_path: Path, cfg: dict):
    out_dir = Path(cfg["output_dir"]) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{pdf_path.name}]  →  {out_dir}")

    from ocr_docling import run_ocr
    results = run_ocr(pdf_path, out_dir, docling_cfg=cfg.get("docling", {}))

    ocr_path = out_dir / "ocr_docling.json"
    ocr_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = out_dir / "text_docling.txt"
    sections = [f"=== Page {r['page_index']} ===\n{r['full_text']}" for r in results]
    text_path.write_text("\n\n".join(sections), encoding="utf-8")

    print(f"  Saved: {ocr_path.name}, {text_path.name}")
    print(f"  Done → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: config.yaml alongside run.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        metavar="STEM",
        help="Skip PDFs whose stem sorts before this value (case-insensitive, e.g. 16RA1717)",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        metavar="PATH",
        help="Path to a .txt file listing PDF filenames (or stems), one per line, "
             "to process from pdf_input. Overrides --start-from.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    start_from = args.start_from or cfg.get("start_from")
    file_list = args.file_list or cfg.get("file_list")
    if file_list and not Path(file_list).is_absolute():
        # Resolve relative to the config file's directory
        file_list = str((Path(args.config).parent / file_list).resolve())
    skip_existing = cfg.get("skip_existing", False)

    pdfs = collect_pdfs(cfg["pdf_input"], start_from=start_from, file_list=file_list)

    if file_list:
        print(f"Using file list: {file_list}  ({len(pdfs)} PDF(s) found)")
    elif start_from:
        print(f"Starting from: {start_from}  ({len(pdfs)} PDF(s) remaining)")
    else:
        print(f"Found {len(pdfs)} PDF(s)")

    if skip_existing:
        before = len(pdfs)
        pdfs = [p for p in pdfs if not is_already_done(p, cfg)]
        skipped = before - len(pdfs)
        if skipped:
            print(f"Skipping {skipped} already-completed PDF(s); {len(pdfs)} remaining")

    for pdf_path in pdfs:
        process_pdf(pdf_path, cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()