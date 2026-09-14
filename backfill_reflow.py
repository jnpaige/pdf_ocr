#!/usr/bin/env python3
"""
backfill_reflow.py — repair the reading order of pages that went through
Docling's raw-OCR fallback, without re-running OCR.

Why this can be a backfill rather than a reprocess: the repair works entirely
from the line bounding boxes already stored in ocr_docling.json. Surya read
these pages correctly — the words are right — and only their ORDER is wrong,
because the fallback path sorted cells into 15pt horizontal bands and then
left-to-right, which on a two-column page alternates between the columns line
by line. Recovering the order needs geometry, not another look at the page.

What it rewrites, for fallback pages only:
    ocr_docling.json  text_lines become paragraph-level entries (text joined,
                      bbox the union, bboxes the per-line regions), and
                      md_text/full_text carry properly ordered paragraphs
                      instead of one fenced block of interleaved lines
    <stem>.md         reassembled from the corrected md_text
    text_docling.txt  reassembled from the corrected full_text

Pages Docling structured successfully are left completely untouched.

Usage:
    uv run python backfill_reflow.py --input-dir "path/to/corpus"
    uv run python backfill_reflow.py --input-dir ... --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from page_reflow import reflow, find_gutters, count_column_switches

BACKUP_SUFFIX = ".pre_reflow.bak"
FALLBACK_MARKER = "Raw OCR text"
# Kept on reflowed pages so a reader still knows Docling did not structure
# this page, even though its text now reads in the right order.
NOTE = ("*Layout not recovered by the document model; reading order and "
        "paragraphs reconstructed from line geometry.*")


def union_bbox(boxes):
    usable = [b for b in boxes if b and b.get("origin") == "BOTTOMLEFT"]
    if not usable:
        return {}
    return {"l": min(b["l"] for b in usable), "t": max(b["t"] for b in usable),
            "r": max(b["r"] for b in usable), "b": min(b["b"] for b in usable),
            "origin": "BOTTOMLEFT"}


def reflow_document(doc_dir: Path, dry_run: bool) -> dict | None:
    ocr_path = doc_dir / "ocr_docling.json"
    if not ocr_path.exists():
        return None
    pages = json.loads(ocr_path.read_text(encoding="utf-8"))
    targets = [p for p in pages if FALLBACK_MARKER in (p.get("md_text") or "")]
    if not targets:
        return None

    stats = dict(document=doc_dir.name, pages=len(targets), lines=0, paragraphs=0,
                 switches_before=0, switches_after=0, joins=0, multi_column=0)

    for page in targets:
        lines = [t for t in page["text_lines"]
                 if t.get("bbox") and (t.get("text") or "").strip()]
        if len(lines) < 4:
            continue
        gutters, x0, x1 = find_gutters(lines)
        paragraphs, _, ordered, joins = reflow(lines)
        if not paragraphs:
            continue

        stats["lines"] += len(lines)
        stats["paragraphs"] += len(paragraphs)
        stats["joins"] += joins
        stats["multi_column"] += 1 if gutters else 0
        stats["switches_before"] += count_column_switches(lines, gutters, x0, x1)
        stats["switches_after"] += count_column_switches(ordered, gutters, x0, x1)

        if dry_run:
            continue

        page["text_lines"] = [{
            "text": para["text"],
            "confidence": min((l.get("confidence", 1.0) for l in para["lines"]), default=1.0),
            "bbox": union_bbox([l["bbox"] for l in para["lines"]]),
            "bboxes": [{**l["bbox"], "page": page["page_index"]} for l in para["lines"]],
        } for para in paragraphs]
        body = "\n\n".join(p["text"] for p in paragraphs)
        page["md_text"] = f"{NOTE}\n\n{body}"
        page["full_text"] = body

    if dry_run:
        return stats

    backup = ocr_path.with_suffix(ocr_path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(ocr_path, backup)
    ocr_path.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")

    md_path = next((p for p in doc_dir.glob("*.md")), None)
    if md_path:
        md_backup = md_path.with_suffix(md_path.suffix + BACKUP_SUFFIX)
        if not md_backup.exists():
            shutil.copy2(md_path, md_backup)
        md_path.write_text(
            "\n\n".join(f"=== Page {p['page_index']} ===\n{p['md_text']}" for p in pages),
            encoding="utf-8")

    txt_path = doc_dir / "text_docling.txt"
    if txt_path.exists():
        txt_backup = txt_path.with_suffix(txt_path.suffix + BACKUP_SUFFIX)
        if not txt_backup.exists():
            shutil.copy2(txt_path, txt_backup)
        txt_path.write_text(
            "\n\n".join(f"=== Page {p['page_index']} ===\n{p.get('full_text','')}" for p in pages),
            encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.input_dir)
    docs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name[0].isdigit())
    print(f"{'DRY RUN — nothing written' if args.dry_run else 'REWRITING'}   {root.name}\n")
    print(f"{'document':44s} {'pages':>6s} {'lines':>6s} {'paras':>6s} "
          f"{'col switches':>16s} {'hyph':>5s}")

    totals = dict(pages=0, lines=0, paragraphs=0, switches_before=0, switches_after=0, joins=0)
    for doc in docs:
        stats = reflow_document(doc, args.dry_run)
        if not stats:
            continue
        print(f"{stats['document'][:43]:44s} {stats['pages']:6d} {stats['lines']:6d} "
              f"{stats['paragraphs']:6d} {stats['switches_before']:6d} -> "
              f"{stats['switches_after']:<6d} {stats['joins']:5d}")
        for k in totals:
            totals[k] += stats[k]

    print(f"\n{totals['pages']} fallback pages repaired across {len(docs)} documents")
    print(f"reading-order column switches: {totals['switches_before']} -> {totals['switches_after']}")
    print(f"{totals['lines']} lines -> {totals['paragraphs']} paragraphs, "
          f"{totals['joins']} hyphenated words rejoined")


if __name__ == "__main__":
    main()
