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


def _report_stem(pdf_path: Path) -> str:
    """Return the report folder name for this PDF, stripping a trailing _ocr suffix.

    When backfilling from existing *_ocr.pdf outputs (from_ocr_pdf mode), the
    input filename is e.g. "22-4793_Gray 2014_ocr.pdf".  Stripping "_ocr" gives
    the canonical report stem used for the output subfolder and .md filename.
    """
    stem = pdf_path.stem
    return stem[:-4] if stem.endswith("_ocr") else stem


def collect_pdfs(
    pdf_input: str,
    start_from: str | None = None,
    file_list: str | None = None,
    from_ocr_pdf: bool = False,
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
            # Strip .pdf suffix if present, but don't use Path.stem — it would
            # split on any dot in the filename (e.g. "et al. 2002" → "et al")
            stem = name[:-4] if name.lower().endswith(".pdf") else name
            candidate = p / f"{stem}.pdf"
            if candidate.exists():
                pdfs.append(candidate)
            else:
                print(f"  WARNING: listed file not found, skipping: {candidate}")
        return pdfs

    if from_ocr_pdf:
        # Backfill mode: find *_ocr.pdf inside immediate subfolders.
        # Each subfolder is one report: <report_name>/<report_name>_ocr.pdf
        pdfs = sorted(p.glob("*/*_ocr.pdf"), key=lambda x: x.parent.name.upper())
        return pdfs

    pdfs = sorted(p.glob("*.pdf"), key=lambda x: x.stem.upper())
    if start_from:
        cutoff = start_from.upper()
        pdfs = [f for f in pdfs if f.stem.upper() >= cutoff]
    return pdfs


def is_already_done(pdf_path: Path, cfg: dict) -> bool:
    out_dir = Path(cfg["output_dir"]) / _report_stem(pdf_path)
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
    from ocr_docling import _stamp_page_label

    merged = fitz.open()
    for cp in chunk_ocr_pdfs:
        if cp.exists():
            sub = fitz.open(str(cp))
            merged.insert_pdf(sub)
            sub.close()

    # Each chunk was built with stamp_page_labels=False (its own page 0 isn't
    # the merged document's page 0), so stamp the real global page index now
    # that every chunk is concatenated into one page sequence — matching what
    # text_docling.txt/.md already use for this document.
    for page_idx in range(len(merged)):
        _stamp_page_label(merged[page_idx], page_idx)

    merged.save(str(out_path))
    merged.close()


def _load_chunk_results(chunk_out: Path, force_reprocess: bool) -> list[dict] | None:
    """Return saved results for a chunk if it completed previously, else None.

    This cache exists so a crashed run can resume mid-document without
    redoing already-completed chunks — that's independent of skip_existing
    (which only gates whether a fully-completed document is reprocessed at
    all) and should keep working normally regardless of it. force_reprocess
    is the deliberate override for "the OCR code itself changed, so even a
    previously-cached chunk needs to actually re-run" — without it, a config
    change like skip_existing: false intended to force a full corpus
    reprocess would silently keep serving every already-chunked document's
    stale per-chunk results instead.
    """
    if force_reprocess:
        return None
    results_path = chunk_out / "ocr_docling.json"
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return None


def process_pdf_chunked(pdf_path: Path, cfg: dict, chunk_size: int, converter):
    from ocr_docling import run_ocr

    stem = _report_stem(pdf_path)
    out_dir = Path(cfg["output_dir"]) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{pdf_path.name}]  →  {out_dir}")

    tmp_dir = out_dir / "_chunks"
    tmp_dir.mkdir(exist_ok=True)

    chunks = _split_pdf(pdf_path, chunk_size, tmp_dir)
    print(f"  Split into {len(chunks)} chunk(s) of up to {chunk_size} pages")

    # Strip markdown_dir — we write the merged .md to the flat dir ourselves
    chunk_docling_cfg = {k: v for k, v in cfg.get("docling", {}).items() if k != "markdown_dir"}

    all_results: list[dict] = []
    chunk_ocr_pdfs: list[Path] = []
    do_ocr = cfg.get("docling", {}).get("do_ocr", True)
    force_reprocess = cfg.get("force_reprocess", False)

    for chunk_path, start_page in chunks:
        chunk_stem = chunk_path.stem
        chunk_out = tmp_dir / chunk_stem
        chunk_out.mkdir(exist_ok=True)

        # Resume: skip chunks that already finished (unless force_reprocess)
        cached = _load_chunk_results(chunk_out, force_reprocess)
        if cached is not None:
            print(f"  Resuming — chunk {chunk_stem} already done, loading cached results")
            chunk_results = cached
        else:
            end_page = start_page + chunk_size - 1
            print(f"  Processing pages {start_page}–{end_page} ({chunk_stem})...")
            chunk_results = run_ocr(chunk_path, chunk_out, docling_cfg=chunk_docling_cfg, converter=converter,
                                     stamp_page_labels=False)
            # Cache chunk results so a future resume can skip this chunk
            (chunk_out / "ocr_docling.json").write_text(
                json.dumps(chunk_results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Offset page indices to match position in the full document
        for r in chunk_results:
            r["page_index"] += start_page
        all_results.extend(chunk_results)

        chunk_ocr_pdfs.append(chunk_out / f"{chunk_stem}_ocr.pdf")

    # --- Merge JSON ---
    ocr_path = out_dir / "ocr_docling.json"
    ocr_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Merge plain text ---
    text_path = out_dir / "text_docling.txt"
    sections = [f"=== Page {r['page_index']} ===\n{r['full_text']}" for r in all_results]
    text_path.write_text("\n\n".join(sections), encoding="utf-8")

    # --- Merge markdown --- (same page_index-offset pattern as text_docling.txt,
    # not a re-read/concat of each chunk's own .md — that would carry each
    # chunk's LOCAL page numbers instead of the corpus-global ones)
    md_sections = [f"=== Page {r['page_index']} ===\n{r['md_text']}" for r in all_results]
    md_content = "\n\n".join(md_sections)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(md_content, encoding="utf-8")

    flat_md_dir = cfg.get("docling", {}).get("markdown_dir")
    if flat_md_dir:
        flat_dir = Path(flat_md_dir)
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"{stem}.md").write_text(md_content, encoding="utf-8")
        print(f"  Mirrored markdown → {flat_dir / stem}.md")

    # --- Merge searchable PDF ---
    if do_ocr:
        merged_pdf_path = out_dir / f"{stem}_ocr.pdf"
        _merge_searchable_pdfs(chunk_ocr_pdfs, merged_pdf_path)
        print(f"  Merged searchable PDF → {merged_pdf_path.name}")

    print(f"  Saved: {ocr_path.name}, {text_path.name}, {md_path.name}")
    print(f"  Done → {out_dir}")


# ---------------------------------------------------------------------------
# Single-PDF processing (unchanged path for small PDFs)
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, cfg: dict, converter):
    chunk_size = cfg.get("chunk_size")
    if chunk_size:
        import fitz
        src = fitz.open(str(pdf_path))
        n_pages = len(src)
        src.close()
        if n_pages > chunk_size:
            print(f"  PDF has {n_pages} pages — chunking into {chunk_size}-page pieces")
            process_pdf_chunked(pdf_path, cfg, chunk_size, converter)
            return

    stem = _report_stem(pdf_path)
    out_dir = Path(cfg["output_dir"]) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{pdf_path.name}]  →  {out_dir}")

    from ocr_docling import run_ocr
    results = run_ocr(pdf_path, out_dir, docling_cfg=cfg.get("docling", {}), stem=stem, converter=converter)

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
    # force_reprocess always wins over skip_existing — it means "the OCR code
    # changed, redo everything," which is a stronger claim than "don't skip
    # already-done documents" and should override it if both are set.
    skip_existing = cfg.get("skip_existing", False) and not cfg.get("force_reprocess", False)
    from_ocr_pdf = cfg.get("from_ocr_pdf", False)

    pdfs = collect_pdfs(cfg["pdf_input"], start_from=start_from, file_list=file_list, from_ocr_pdf=from_ocr_pdf)

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

    # Built once and reused for every PDF in this run — see build_converter's
    # docstring for why rebuilding it per-PDF caused memory growth on large
    # corpus runs.
    from ocr_docling import build_converter
    converter = build_converter(cfg.get("docling", {}))

    failed: list[str] = []
    for pdf_path in pdfs:
        try:
            process_pdf(pdf_path, cfg, converter)
        except Exception as e:
            # One malformed/unusual PDF must not take the rest of a
            # corpus-scale run down with it — log and move on. Common real
            # cause: Docling's backend rejecting a filename/PDF structure
            # that other tools (PyMuPDF) read fine; see _safe_convert_path
            # in lib/ocr_docling.py for the filename-encoding case that's
            # already handled — this catches whatever isn't.
            print(f"  [ERROR] {pdf_path.name}: {type(e).__name__}: {e}")
            failed.append(pdf_path.name)

    if failed:
        print(f"\n{len(failed)} PDF(s) failed and were skipped:")
        for name in failed:
            print(f"  - {name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()