"""
backfill_txt.py — Rebuild text_docling.txt with heading markers from existing outputs.

Prior versions of pdf_ocr wrote text_docling.txt with page delimiters only.
The .md file had heading markup (## Section) but no page markers.
This script merges them: it reads heading strings + depths from the .md,
matches them against the text_lines in ocr_docling.json, and writes a new
text_docling.txt where headings are prefixed with the correct # markers.

No neural inference is run — this is a pure text operation.

Usage:
    # Single report folder:
    python backfill_txt.py --report-dir "D:/Local_repository/phase_ii_reports_ocr_docling/22-4793_Gray 2014"

    # Flat directory of report folders:
    python backfill_txt.py --dir "D:/Local_repository/phase_ii_reports_ocr_docling"

    # District-subfoldered directory (e.g. GDrive layout):
    python backfill_txt.py --dir "G:/Shared drives/.../Phase_II_Reports_ocr_docling" --subfolders

    # Dry run (print what would change, write nothing):
    python backfill_txt.py --dir "..." --dry-run
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path


def _unescape_md(text: str) -> str:
    """Reverse markdown escaping applied by docling's serializer."""
    text = html.unescape(text)
    text = re.sub(r'\\([_*`\[\]()#+\-.!|])', r'\1', text)
    return text


def _normalize(text: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', text).strip()


def parse_headings(md_path: Path) -> dict[str, int]:
    """Return {raw_heading_text: num_hashes} from a docling-generated .md file."""
    headings: dict[str, int] = {}
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            num_hashes = len(m.group(1))
            raw = _unescape_md(m.group(2).strip())
            headings[_normalize(raw)] = num_hashes
    return headings


def rebuild_report(report_dir: Path, dry_run: bool = False) -> bool:
    """Rebuild text_docling.txt for one report folder. Returns True if written."""
    stem = report_dir.name
    md_path   = report_dir / f"{stem}.md"
    json_path = report_dir / "ocr_docling.json"
    txt_path  = report_dir / "text_docling.txt"

    if not md_path.exists():
        print(f"  SKIP {stem} — no .md file")
        return False
    if not json_path.exists():
        print(f"  SKIP {stem} — no ocr_docling.json")
        return False

    headings = parse_headings(md_path)
    if not headings:
        print(f"  SKIP {stem} — no headings found in .md")
        return False

    pages = json.loads(json_path.read_text(encoding="utf-8"))

    sections: list[str] = []
    heading_hits = 0

    for page in pages:
        lines: list[str] = []
        for entry in page.get("text_lines", []):
            text = entry.get("text", "")
            key = _normalize(text)
            if key in headings:
                num_hashes = headings[key]
                text = f"{'#' * num_hashes} {text}"
                heading_hits += 1
            lines.append(text)
        full_text = "\n".join(lines)
        sections.append(f"=== Page {page['page_index']} ===\n{full_text}")

    new_content = "\n\n".join(sections)

    if dry_run:
        print(f"  DRY RUN {stem} — {len(pages)} pages, {heading_hits} heading matches")
        return False

    txt_path.write_text(new_content, encoding="utf-8")
    print(f"  OK  {stem} — {len(pages)} pages, {heading_hits} heading instances marked")
    return True


def collect_report_dirs(base: Path, subfolders: bool) -> list[Path]:
    """Collect report directories to process."""
    if subfolders:
        # GDrive layout: base/<district>/<report>/
        # Skip *_md companion directories
        dirs = []
        for district in sorted(base.iterdir()):
            if not district.is_dir() or district.name.endswith("_md"):
                continue
            for report in sorted(district.iterdir()):
                if report.is_dir() and (report / "ocr_docling.json").exists():
                    dirs.append(report)
        return dirs
    else:
        # Flat layout: base/<report>/
        return sorted(
            d for d in base.iterdir()
            if d.is_dir() and (d / "ocr_docling.json").exists()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report-dir", metavar="PATH", help="Single report folder to process")
    group.add_argument("--dir", metavar="PATH", help="Directory containing report folders")
    parser.add_argument("--subfolders", action="store_true",
                        help="When using --dir: expect district subfolders (GDrive layout)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing anything")
    args = parser.parse_args()

    if args.report_dir:
        report_dir = Path(args.report_dir)
        if not report_dir.is_dir():
            print(f"ERROR: not a directory: {report_dir}", file=sys.stderr)
            sys.exit(1)
        rebuild_report(report_dir, dry_run=args.dry_run)
    else:
        base = Path(args.dir)
        if not base.is_dir():
            print(f"ERROR: not a directory: {base}", file=sys.stderr)
            sys.exit(1)
        dirs = collect_report_dirs(base, subfolders=args.subfolders)
        print(f"Found {len(dirs)} report(s) to process")
        written = sum(1 for d in dirs if rebuild_report(d, dry_run=args.dry_run))
        action = "Would update" if args.dry_run else "Updated"
        print(f"\n{action} {written}/{len(dirs)} text_docling.txt files")


if __name__ == "__main__":
    main()
