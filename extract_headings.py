#!/usr/bin/env python3
"""
extract_headings.py — Post-hoc heading extraction from existing text_docling.txt files.

Walks a directory tree, finds every text_docling.txt that doesn't already have
a headings.json alongside it, extracts headings, and writes headings.json
in the same directory.

Usage:
    uv run python extract_headings.py --input-dir "G:/path/to/ocr_output"
    uv run python extract_headings.py --input-dir "G:/path/to/ocr_output" --force
    uv run python extract_headings.py --input-dir "G:/path/to/ocr_output" --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from ocr_docling import write_headings_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract headings from text_docling.txt files")
    ap.add_argument("--input-dir", required=True, metavar="PATH",
                    help="Root directory to search for text_docling.txt files")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if headings.json already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be done without writing any files")
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    txt_files = sorted(root.rglob("text_docling.txt"))
    if not txt_files:
        sys.exit(f"No text_docling.txt files found under: {root}")

    print(f"Found {len(txt_files)} text_docling.txt file(s) under {root}\n")

    n_written = 0
    n_skipped = 0

    for txt_path in txt_files:
        out_dir = txt_path.parent
        h_path  = out_dir / "headings.json"

        if h_path.exists() and not args.force:
            print(f"  [skip] {out_dir.name}")
            n_skipped += 1
            continue

        doc_name = out_dir.name

        if args.dry_run:
            print(f"  [would write] {h_path}")
            n_written += 1
            continue

        h_path = write_headings_json(txt_path, out_dir, doc_name)
        import json
        n = json.loads(h_path.read_text(encoding="utf-8")).get("n_headings", 0)
        print(f"  [done] {doc_name}  ({n} headings)")
        n_written += 1

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Done.  "
          f"{n_written} written, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
