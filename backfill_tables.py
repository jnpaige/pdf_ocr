#!/usr/bin/env python3
"""
backfill_tables.py — add table geometry to an ocr_docling.json that was written
before tables were exported, without re-running OCR.

Tables used to be dropped from ocr_docling.json entirely: a docling TableItem
has no `.text`, so the text-item loop skipped it before reading its provenance,
and a table survived only as pipe rows in md_text with no coordinates. That is
fixed for new runs, but every already-processed corpus still has no table
geometry — and reprocessing the Kisatchie corpora is a multi-day job.

It doesn't need one. Table structure comes from docling's layout/TableFormer
models reading the page, NOT from the OCR text layer, so running a layout-only
pass (do_ocr=False) over the existing <stem>_ocr.pdf recovers the same geometry
a full reprocess would. Only the `tables` key is written; `text_lines`,
`full_text` and `md_text` are left exactly as Surya produced them, so the .md
every downstream segment/tag file was derived from stays valid and nothing
downstream needs re-running.

What this does NOT backfill: the per-item `bboxes` list (multi-region
provenance for paragraphs spanning columns/pages). That describes text items,
which this pass would re-derive differently from Surya's, so it genuinely
requires reprocessing. Documents keep first-region-only geometry for prose
until then.

Usage:
    uv run python backfill_tables.py --input-dir "path/to/corpus"
    uv run python backfill_tables.py --doc-dir "path/to/<document folder>"
    uv run python backfill_tables.py --input-dir ... --no-backup --force
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

BACKUP_SUFFIX = ".pre_tables.bak"


def _build_converter():
    """One converter for the whole run — rebuilding it per PDF was a real
    source of memory growth on long runs in this repo's history."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()
    opts.do_ocr = False           # the text layer is already there; we want layout only
    opts.do_table_structure = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def backfill_document(doc_dir: Path, converter, backup: bool, force: bool) -> tuple[int, int] | None:
    from docling_core.types.doc.document import TableItem
    import ocr_docling as O

    stem = doc_dir.name
    ocr_json_path = doc_dir / "ocr_docling.json"
    pdf_path = doc_dir / f"{stem}_ocr.pdf"

    if not ocr_json_path.exists():
        print(f"  SKIP {stem} — no ocr_docling.json")
        return None
    if not pdf_path.exists():
        print(f"  SKIP {stem} — no {pdf_path.name} to read layout from")
        return None

    pages = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    if not force and any(p.get("tables") for p in pages):
        print(f"  SKIP {stem} — already has table geometry (use --force to redo)")
        return None

    doc = converter.convert(str(pdf_path)).document
    heights = O._page_heights(doc)

    tables_by_page: dict[int, list[dict]] = defaultdict(list)
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            prov = getattr(item, "prov", None)
            page_index = (prov[0].page_no - 1) if prov else 0
            tables_by_page[page_index].append(O._table_to_dict(item, heights))

    # A page-count mismatch means the layout pass and the stored OCR disagree
    # about the document — merging by page index would put tables on the wrong
    # pages, so refuse rather than corrupt the file.
    n_layout_pages = max(max(tables_by_page, default=-1) + 1, len(heights))
    if n_layout_pages > len(pages):
        print(f"  SKIP {stem} — layout pass saw {n_layout_pages} pages, "
              f"ocr_docling.json has {len(pages)}")
        return None

    text_before = sum(len(p.get("text_lines", [])) for p in pages)
    for page in pages:
        page["tables"] = tables_by_page.get(page.get("page_index", 0), [])
    text_after = sum(len(p.get("text_lines", [])) for p in pages)
    assert text_before == text_after, "text_lines must not change"

    if backup:
        backup_path = ocr_json_path.with_suffix(ocr_json_path.suffix + BACKUP_SUFFIX)
        if not backup_path.exists():
            shutil.copy2(ocr_json_path, backup_path)

    ocr_json_path.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")

    n_tables = sum(len(v) for v in tables_by_page.values())
    n_rows = sum(len([r for r in t.get("rows", []) if r.get("bbox")])
                 for v in tables_by_page.values() for t in v)
    print(f"  {stem}: {n_tables} table(s), {n_rows} rows with geometry "
          f"({text_before} text_lines unchanged)")
    return n_tables, n_rows


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-dir")
    g.add_argument("--input-dir")
    ap.add_argument("--no-backup", action="store_true",
                    help=f"Don't keep an ocr_docling.json{BACKUP_SUFFIX} copy")
    ap.add_argument("--force", action="store_true",
                    help="Redo documents that already carry table geometry")
    args = ap.parse_args()

    if args.doc_dir:
        dirs = [Path(args.doc_dir)]
    else:
        root = Path(args.input_dir)
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "ocr_docling.json").exists())
    if not dirs:
        sys.exit("No documents with an ocr_docling.json found")

    print(f"Backfilling table geometry into {len(dirs)} document(s)\n")
    converter = _build_converter()

    done = tables = rows = 0
    for d in dirs:
        result = backfill_document(d, converter, backup=not args.no_backup, force=args.force)
        if result:
            done += 1
            tables += result[0]
            rows += result[1]
    print(f"\n{done} document(s) updated — {tables} tables, {rows} rows with geometry")


if __name__ == "__main__":
    main()
