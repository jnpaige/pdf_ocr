"""
fix_rotated_pages.py — find pages that were scanned sideways, read them again turned upright, and patch them
into a finished pdf_ocr volume.

Some scanned volumes contain pages whose image is turned 90° inside an ordinary portrait page (no /Rotate flag).
Surya then reads the text as many short vertical boxes, and the OCR for that page is close to useless. This tool
works on finished output and touches only those pages:

1. Find candidates from the volume's own ocr_docling.json: pages whose `ocr_lines` boxes are mostly taller than wide
   (share > --tall-share, with at least --min-lines lines). No OCR is needed for this step.
2. Build a small PDF of the candidate pages turned upright (--rotate degrees clockwise). Each new page draws the
   original page as a rotated form XObject, so the scan image is reused, not re-encoded, and the page carries no
   /Rotate flag. A pixel check against pdfium's own rotated rendering guards the direction.
3. Read those pages with the same Docling + Surya pipeline and config as run.py, in --chunk-size pieces.
4. Keep a new reading only if it is better (see accept()): mean line confidence higher by at least --margin, or the
   boxes flip decisively from mostly tall to almost none with no loss of confidence or line length. Otherwise the
   page keeps its original reading, so a page that wasn't really sideways is left alone.
5. Patch the volume: replace accepted pages in ocr_docling.json (each gets "rotation_applied": <degrees clockwise>,
   so its boxes can be mapped back to the original scan), rebuild text_docling.txt and <stem>.md from it exactly as
   run.py does, and swap the pages in <stem>_ocr.pdf, re-stamping their `=== Page N ===` labels.

The original four files are copied to --backup-dir/<stem>/ before anything is changed (once; an existing backup is
never overwritten). Patched pages are skipped on a rerun, so the tool is safe to run again. Every candidate page is
logged to --log (CSV) with its old and new scores and whether it was accepted.

Run one volume per process on large corpora (memory is returned between volumes), e.g. from a supervisor loop.

Usage:
    uv run python fix_rotated_pages.py --config config_x.local.yaml --ocr-dir <output root> \
        --pdf-dir <source pdfs> [--pdf-dir <more>] --backup-dir <folder> --file-list volumes.txt [--dry-run]
"""

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "lib"))

LOG_FIELDS = ["time", "volume", "page", "rotate_cw", "old_lines", "old_tall", "old_conf", "old_median_len",
              "new_lines", "new_tall", "new_conf", "new_median_len", "accepted"]


def page_stats(page: dict) -> dict:
    """Line count, share of boxes taller than wide, mean confidence and median length of a page's ocr_lines."""
    lines = [l for l in page.get("ocr_lines") or [] if (l.get("text") or "").strip()]
    if not lines:
        return {"lines": 0, "tall": 0.0, "conf": 0.0, "median_len": 0}
    tall = sum(1 for l in lines if (l["bbox"]["t"] - l["bbox"]["b"]) > (l["bbox"]["r"] - l["bbox"]["l"]))
    return {"lines": len(lines), "tall": round(tall / len(lines), 3),
            "conf": round(statistics.mean(l.get("confidence") or 0.0 for l in lines), 4),
            "median_len": statistics.median(len(l["text"].strip()) for l in lines)}


def find_candidates(results: list[dict], tall_share: float, min_lines: int) -> list[int]:
    out = []
    for r in results:
        if r.get("rotation_applied"):
            continue
        s = page_stats(r)
        if s["lines"] >= min_lines and s["tall"] > tall_share:
            out.append(r["page_index"])
    return out


def accept(old: dict, new: dict, margin: float) -> bool:
    """Keep the upright reading if it is clearly better.

    Either confidence rises by at least `margin` (with fewer tall boxes), or the geometry flips decisively — mostly
    tall boxes before, almost none after — with lines no shorter and confidence within `margin`. The second case
    covers pages Surya had already read passably sideways (it turns tall line crops itself), where confidence barely
    moves but the lines stop being fragments.
    """
    if new["lines"] == 0:
        return False
    gained = new["conf"] >= old["conf"] + margin and new["tall"] < old["tall"]
    flipped = (old["tall"] >= 0.5 and new["tall"] <= 0.1 and new["median_len"] >= old["median_len"]
               and new["conf"] >= old["conf"] - margin)
    return gained or flipped


def build_upright_pdf(src_pdf: Path, pages: list[int], rotate_cw: int, out_path: Path) -> None:
    """Write the given pages, turned rotate_cw degrees clockwise, as a new PDF (lossless; no /Rotate flag)."""
    import fitz

    src = fitz.open(str(src_pdf))
    dst = fitz.open()
    for p in pages:
        r = src[p].rect
        w, h = (r.height, r.width) if rotate_cw % 180 else (r.width, r.height)
        page = dst.new_page(width=w, height=h)
        # show_pdf_page turns counter-clockwise for positive angles
        page.show_pdf_page(page.rect, src, p, rotate=-rotate_cw % 360)
    dst.save(str(out_path), garbage=3, deflate=True)
    dst.close()
    src.close()


def check_direction(src_pdf: Path, page: int, upright_pdf: Path, rotate_cw: int) -> float:
    """Mean absolute pixel difference between our upright page and pdfium's own rotated render (0 = identical)."""
    import numpy as np
    import pypdfium2 as pdfium

    a = pdfium.PdfDocument(str(src_pdf))[page].render(scale=0.25, rotation=rotate_cw).to_pil().convert("L")
    b = pdfium.PdfDocument(str(upright_pdf))[0].render(scale=0.25).to_pil().convert("L")
    if a.size != b.size:
        b = b.resize(a.size)
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def write_volume_outputs(out_dir: Path, stem: str, results: list[dict]) -> None:
    """Rewrite ocr_docling.json, text_docling.txt and <stem>.md from results, the same way run.py merges chunks."""
    (out_dir / "ocr_docling.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "text_docling.txt").write_text(
        "\n\n".join(f"=== Page {r['page_index']} ===\n{r['full_text']}" for r in results), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(
        "\n\n".join(f"=== Page {r['page_index']} ===\n{r['md_text']}" for r in results), encoding="utf-8")


def patch_searchable_pdf(vol_pdf: Path, swaps: list[tuple[int, Path, int]]) -> None:
    """Replace pages of the volume's _ocr.pdf: swaps = [(volume page, patch _ocr.pdf, page in that pdf)]."""
    import fitz
    from ocr_docling import _stamp_page_label

    doc = fitz.open(str(vol_pdf))
    n = len(doc)
    patches: dict[Path, "fitz.Document"] = {}
    for vp, ppdf, pp in sorted(swaps):
        if ppdf not in patches:
            patches[ppdf] = fitz.open(str(ppdf))
        doc.insert_pdf(patches[ppdf], from_page=pp, to_page=pp, start_at=vp)  # new page lands at vp
        doc.delete_page(vp + 1)                                                 # old page moved to vp + 1
        _stamp_page_label(doc[vp], vp)
    assert len(doc) == n, f"page count changed: {n} -> {len(doc)}"
    tmp = vol_pdf.with_suffix(".pdf.partial")
    doc.save(str(tmp), garbage=3, deflate=True)
    doc.close()
    for d in patches.values():
        d.close()
    tmp.replace(vol_pdf)


def fix_volume(pdf_path: Path, out_dir: Path, args, cfg: dict, converter, log: csv.DictWriter) -> dict:
    from ocr_docling import run_ocr

    stem = out_dir.name
    results = json.loads((out_dir / "ocr_docling.json").read_text(encoding="utf-8"))
    cands = find_candidates(results, args.tall_share, args.min_lines)
    summary = {"volume": stem, "candidates": len(cands), "accepted": 0}
    print(f"\n[{stem}] {len(results)} pages, {len(cands)} sideways candidate(s)")
    if not cands or args.dry_run:
        return summary

    backup = Path(args.backup_dir) / stem
    if not backup.exists():
        backup.mkdir(parents=True)
        for name in ("ocr_docling.json", "text_docling.txt", f"{stem}.md", f"{stem}_ocr.pdf"):
            shutil.copy2(out_dir / name, backup / name)
        print(f"  Backed up originals → {backup}")

    docling_cfg = {k: v for k, v in cfg.get("docling", {}).items() if k != "markdown_dir"}
    by_index = {r["page_index"]: r for r in results}
    swaps: list[tuple[int, Path, int]] = []
    with tempfile.TemporaryDirectory(prefix="rot_", dir=args.work_dir) as tmp:
        tmp = Path(tmp)
        for k in range(0, len(cands), args.chunk_size):
            part = cands[k:k + args.chunk_size]
            part_dir = tmp / f"part_{k:05d}"
            part_dir.mkdir()
            part_pdf = part_dir / f"part_{k:05d}.pdf"
            build_upright_pdf(pdf_path, part, args.rotate, part_pdf)
            if k == 0:
                probe = part_dir / "probe.pdf"
                build_upright_pdf(pdf_path, part[:1], args.rotate, probe)
                diff = check_direction(pdf_path, part[0], probe, args.rotate)
                if diff > 8:
                    raise RuntimeError(f"upright page does not match pdfium's {args.rotate}° render (diff {diff:.1f})")
            print(f"  Reading {len(part)} page(s) turned {args.rotate}° clockwise...")
            new = run_ocr(part_pdf, part_dir, docling_cfg=docling_cfg, converter=converter, stamp_page_labels=False)
            for j, (vp, nr) in enumerate(zip(part, new)):
                old_s, new_s = page_stats(by_index[vp]), page_stats(nr)
                ok = accept(old_s, new_s, args.margin)
                log.writerow({"time": datetime.now().isoformat(timespec="seconds"), "volume": stem, "page": vp,
                              "rotate_cw": args.rotate,
                              **{f"old_{x}": old_s[x] for x in ("lines", "tall", "conf", "median_len")},
                              **{f"new_{x}": new_s[x] for x in ("lines", "tall", "conf", "median_len")},
                              "accepted": int(ok)})
                if ok:
                    nr["page_index"] = vp
                    nr["rotation_applied"] = args.rotate
                    by_index[vp] = nr
                    swaps.append((vp, part_dir / f"{part_pdf.stem}_ocr.pdf", j))
        summary["accepted"] = len(swaps)
        if swaps:
            results = [by_index[i] for i in sorted(by_index)]
            patch_searchable_pdf(out_dir / f"{stem}_ocr.pdf", swaps)
            write_volume_outputs(out_dir, stem, results)
    print(f"  Accepted {summary['accepted']} of {len(cands)}"
          + ("; volume outputs patched" if summary["accepted"] else "; volume unchanged"))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="pdf_ocr config used for the original run (docling settings, "
                                                     "recognition_checkpoint, chunk_size)")
    ap.add_argument("--ocr-dir", required=True, help="output root holding one folder per volume")
    ap.add_argument("--pdf-dir", action="append", required=True, help="folder of source PDFs (repeatable)")
    ap.add_argument("--file-list", help="volumes to check, one PDF name or stem per line (default: every folder)")
    ap.add_argument("--backup-dir", required=True, help="where the original four files of a patched volume go")
    ap.add_argument("--log", help="CSV of every candidate page (default: <ocr-dir>/rotation_fixes.csv)")
    ap.add_argument("--work-dir", default=None, help="temp folder for the upright PDFs (default: system temp)")
    ap.add_argument("--rotate", type=int, default=270, choices=(90, 180, 270), help="degrees clockwise (default 270)")
    ap.add_argument("--tall-share", type=float, default=0.15, help="candidate if this share of boxes is tall")
    ap.add_argument("--min-lines", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.03, help="required gain in mean line confidence")
    ap.add_argument("--chunk-size", type=int, default=None, help="pages per OCR call (default: config chunk_size)")
    ap.add_argument("--dry-run", action="store_true", help="only count candidates")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    args.chunk_size = args.chunk_size or cfg.get("chunk_size") or 25
    ocr_dir = Path(args.ocr_dir)
    if args.file_list:
        stems = [n.strip()[:-4] if n.strip().lower().endswith(".pdf") else n.strip()
                 for n in Path(args.file_list).read_text(encoding="utf-8").splitlines() if n.strip()]
    else:
        stems = sorted(d.name for d in ocr_dir.iterdir() if (d / "ocr_docling.json").exists())

    class LazyConverter:  # load the OCR models only when a volume actually has candidates
        def __init__(self):
            self._c = None

        def __getattr__(self, name):
            if self._c is None:
                from ocr_docling import build_converter
                self._c = build_converter(cfg.get("docling", {}))
            return getattr(self._c, name)

    converter = None if args.dry_run else LazyConverter()

    log_path = Path(args.log) if args.log else ocr_dir / "rotation_fixes.csv"
    new_log = not log_path.exists()
    totals, failed = [0, 0], []
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        log = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        for stem in stems:
            pdf = next((Path(d) / f"{stem}.pdf" for d in args.pdf_dir if (Path(d) / f"{stem}.pdf").exists()), None)
            if pdf is None or not (ocr_dir / stem / "ocr_docling.json").exists():
                print(f"  [skip] {stem}: source PDF or OCR output not found")
                failed.append(stem)
                continue
            try:
                s = fix_volume(pdf, ocr_dir / stem, args, cfg, converter, log)
                totals[0] += s["candidates"]
                totals[1] += s["accepted"]
            except Exception as e:
                print(f"  [ERROR] {stem}: {type(e).__name__}: {e}")
                failed.append(stem)
            fh.flush()
    print(f"\n{len(stems)} volume(s): {totals[0]} candidate page(s), {totals[1]} patched"
          + (f"; {len(failed)} failed/skipped: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
