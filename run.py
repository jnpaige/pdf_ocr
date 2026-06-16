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

Chunked processing:
    Set chunk_size in config.yaml to split large PDFs into N-page pieces before
    OCR, then merge the results. Intermediate chunk files are kept under
    <out_dir>/_chunks/ so a crashed run can resume from where it left off.
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


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _split_pdf(pdf_path: Path, chunk_size: int, tmp_dir: Path) -> list[tuple[Path, int]]:
    """Write N-page chunk PDFs to tmp_dir. Returns (chunk_path, start_page) pairs."""
    import fitz

    src = fitz.open(str(pdf_path))
    n = len(src)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(src, from_page=start, to_page=end - 1)
        chunk_path = tmp_dir / f"chunk_{start:05d}.pdf"
        chunk_doc.save(str(chunk_path))
        chunk_doc.close()
        chunks.append((chunk_path, start))
    src.close()
    return chunks


def _merge_searchable_pdfs(chunk_ocr_pdfs: list[Path], out_path: Path) -> None:
    import fitz

    merged = fitz.open()
    for cp in chunk_ocr_pdfs:
        if cp.exists():
            sub = fitz.open(str(cp))
            merged.insert_pdf(sub)
            sub.close()
    merged.save(str(out_path))
    merged.close()


def _load_chunk_results(chunk_out: Path) -> list[dict] | None:
    """Return saved results for a chunk if it completed previously, else None."""
    results_path = chunk_out / "ocr_docling.json"
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return None


def process_pdf_chunked(pdf_path: Path, cfg: dict, chunk_size: int):
    from ocr_docling import run_ocr

    out_dir = Path(cfg["output_dir"]) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{pdf_path.name}]  →  {out_dir}")

    tmp_dir = out_dir / "_chunks"
    tmp_dir.mkdir(exist_ok=True)

    chunks = _split_pdf(pdf_path, chunk_size, tmp_dir)
    print(f"  Split into {len(chunks)} chunk(s) of up to {chunk_size} pages")

    # Strip markdown_dir — we write the merged .md to the flat dir ourselves
    chunk_docling_cfg = {k: v for k, v in cfg.get("docling", {}).items() if k != "markdown_dir"}

    all_results: list[dict] = []
    chunk_mds: list[str] = []
    chunk_ocr_pdfs: list[Path] = []
    do_ocr = cfg.get("docling", {}).get("do_ocr", True)

    for chunk_path, start_page in chunks:
        chunk_stem = chunk_path.stem
        chunk_out = tmp_dir / chunk_stem
        chunk_out.mkdir(exist_ok=True)

        # Resume: skip chunks that already finished
        cached = _load_chunk_results(chunk_out)
        if cached is not None:
            print(f"  Resuming — chunk {chunk_stem} already done, loading cached results")
            chunk_results = cached
        else:
            end_page = start_page + chunk_size - 1
            print(f"  Processing pages {start_page}–{end_page} ({chunk_stem})...")
            chunk_results = run_ocr(chunk_path, chunk_out, docling_cfg=chunk_docling_cfg)
            # Cache chunk results so a future resume can skip this chunk
            (chunk_out / "ocr_docling.json").write_text(
                json.dumps(chunk_results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Offset page indices to match position in the full document
        for r in chunk_results:
            r["page_index"] += start_page
        all_results.extend(chunk_results)

        chunk_md = chunk_out / f"{chunk_stem}.md"
        if chunk_md.exists():
            chunk_mds.append(chunk_md.read_text(encoding="utf-8"))

        chunk_ocr_pdfs.append(chunk_out / f"{chunk_stem}_ocr.pdf")

    # --- Merge JSON ---
    ocr_path = out_dir / "ocr_docling.json"
    ocr_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Merge plain text ---
    text_path = out_dir / "text_docling.txt"
    sections = [f"=== Page {r['page_index']} ===\n{r['full_text']}" for r in all_results]
    text_path.write_text("\n\n".join(sections), encoding="utf-8")

    # --- Merge markdown ---
    md_content = "\n\n".join(chunk_mds)
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")

    flat_md_dir = cfg.get("docling", {}).get("markdown_dir")
    if flat_md_dir:
        flat_dir = Path(flat_md_dir)
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"{pdf_path.stem}.md").write_text(md_content, encoding="utf-8")
        print(f"  Mirrored markdown → {flat_dir / pdf_path.stem}.md")

    # --- Merge searchable PDF ---
    if do_ocr:
        merged_pdf_path = out_dir / f"{pdf_path.stem}_ocr.pdf"
        _merge_searchable_pdfs(chunk_ocr_pdfs, merged_pdf_path)
        print(f"  Merged searchable PDF → {merged_pdf_path.name}")

    print(f"  Saved: {ocr_path.name}, {text_path.name}, {md_path.name}")
    print(f"  Done → {out_dir}")


# ---------------------------------------------------------------------------
# Single-PDF processing (unchanged path for small PDFs)
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, cfg: dict):
    chunk_size = cfg.get("chunk_size")
    if chunk_size:
        import fitz
        src = fitz.open(str(pdf_path))
        n_pages = len(src)
        src.close()
        if n_pages > chunk_size:
            print(f"  PDF has {n_pages} pages — chunking into {chunk_size}-page pieces")
            process_pdf_chunked(pdf_path, cfg, chunk_size)
            return

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