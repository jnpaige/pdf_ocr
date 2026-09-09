#!/usr/bin/env python3
"""
fill_missing_captions.py — recover captions Docling failed to associate with
their figure, using process-of-elimination against a document's own already-
confirmed figure numbers plus a local reasoning-capable LLM.

This is a post-processing pass over pdf_ocr's own finished output (figures/
figures.json + text_docling.txt) — it does not touch Docling or re-run OCR.

Why elimination, not free search: a bare regex or an open "find the caption
for Fig. N" prompt runs into two real problems worked out interactively
against a real corpus (Broich et al. 2021) — (1) regex can't reliably tell
a real caption block apart from an in-text citation of the same figure
without a pile of brittle special cases, and (2) even a targeted LLM prompt
can pick a WRONG real caption when a page has several genuine "Fig. N"
blocks nearby. The fix for both: never treat the caption's printed number
as a free-form guess. For each uncaptioned figure, compute the small set of
numbers it is actually still possible to be — global elimination (every
number already confirmed elsewhere in the document is excluded) intersected
with local document-order bounds (its nearest confirmed neighbors before
and after) — and give the LLM that explicit candidate list rather than one
guess. Verified end-to-end on Broich: with candidates narrowed to a single
number, a smaller model (qwen2.5:14b) still missed captions sitting in
citation-dense OCR text; a reasoning model (qwq:32b) reading the same
constrained prompt recovered all of them, verbatim, every self-reported
number matching its candidate list.

Figures are resolved most-constrained-first within a document: solving the
easy (small-candidate-set) figures first adds them to the confirmed set,
which tightens the candidate pools of the harder ones still to come. A
document with zero Docling-confirmed captions to begin with has no anchor
point for elimination at all and is left untouched rather than guessed at.

On a successful resolution: figures.json's `caption`/`printed_figure_number`
are updated, the already-saved PNG is re-stamped with the corrected
watermark number and gets the caption box appended
(compose_figure_with_caption) — the same finishing step _extract_figures
already applies to figures Docling captioned directly.

Usage:
    uv run python fill_missing_captions.py --config config_fill_captions.yaml
    uv run python fill_missing_captions.py --config config_fill_captions.yaml --stem "Broich et al. (2021) Ifri n'Etsedda"
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from ollama_client import extract_json
from ocr_docling import stamp_figure_label, compose_figure_with_caption, _save_image_with_retry

SYSTEM_PROMPT = """You are extracting a figure caption from OCR'd academic-paper text.

You will be given:
- A short list of POSSIBLE (series, number) candidates for this specific
  figure — determined by process of elimination against every other figure
  already confirmed elsewhere in this document. The real caption for THIS
  figure is guaranteed to match one of these, and cannot be any other
  series+number combination.
- A window of OCR text covering a few consecutive pages of the document.

Some papers (common in Nature-family journals) carry TWO INDEPENDENT figure
numbering sequences that both restart at 1: ordinary main-text captions
("Fig. 3 Selected artifacts..." / "Figure 3 | Selected artifacts...") and a
SEPARATE "Extended Data" sequence ("Extended Data Fig. 3 ..." /
"Extended Data Figure 3 | ..."). These are two different counters — main
Figure 3 and Extended Data Figure 3 are two completely different figures
that both happen to use the number 3. Each possible candidate below is
tagged with its series ("main" or "extended_data") for exactly this reason
— always match BOTH the series and the number, never the number alone.
A caption that starts with "Extended Data Fig." or "Extended Data Figure"
belongs to the extended_data series; a plain "Fig."/"Figure" caption
(NOT preceded by "Extended Data") belongs to the main series.

Some pages in that window may be OCR read directly off a figure image itself
— specimen ID labels, scale bars ("1 cm"), legend symbols, or counts
scattered at their own positions on the page, not connected prose. Such a
page looks like a list of short disconnected fragments (e.g. "INES 9688",
"0", "1 cm") rather than sentences. A real caption is always a grammatical,
self-contained sentence or paragraph — it does not occur inside that kind
of fragment list. If a page in the window looks like this, do not try to
extract meaning from it or search it further for a caption; move on to the
next page in the window instead.

Your task: find the actual printed caption matching ONE of the possible
(series, number) candidates — a self-contained description that STARTS a
block of text with "Fig."/"Figure" or "Extended Data Fig."/"Extended Data
Figure" followed by one of those numbers (e.g. "Fig. 3 Selected artifacts
from..." or "Extended Data Figure 3 | Selected artifacts..."). Do NOT
return a sentence that merely CITES or references a figure in passing
(e.g. "as shown in Fig. 3", "(Fig. 3)", "(Extended Data Fig. 3)") — only
the actual caption block itself, which is normally a longer, self-contained
description of what the figure shows.

If the text contains real captions for (series, number) combinations NOT in
the possible list, ignore them — they belong to a different figure
elsewhere in the document, not this one.

IMPORTANT — more than one possible candidate may have a real caption block
present in this same window, and a single page can contain MORE THAN ONE
caption block interleaved with ordinary body paragraphs (a caption is
often followed or preceded on the same page by unrelated prose that just
happens to mention a different figure in passing — keep scanning the
whole page, don't stop at the first "Fig." you hit on it).

You must build your answer in this order, as the JSON fields below are
ordered — do not decide the final answer first and justify it afterward:

1. First, populate "candidates_found": scan every page in the window, in
   full, and list EVERY real caption block you find whose (series, number)
   is in the possible list — one entry per candidate, with the page it's
   actually on, its series ("main" or "extended_data"), and the caption's
   own printed number. Do this for the whole window before moving on — do
   not stop once you've found one. Double check each page number against
   the "--- Page N ---" marker it actually appeared under; do not guess or
   misremember which page a candidate was on.
2. Then, ONLY from what you listed in step 1 (do not introduce a new
   candidate here), pick the single BEST match for THIS FIGURE, whose own
   page is given below: the candidate whose page is closest to it — a
   candidate on that exact same page (distance 0) always beats one on any
   other page, even the very next one. If more than one candidate ties on
   distance, prefer whichever has the numerically lowest possible number
   within its own series (captions run in increasing order through a
   document, so the earliest un-eliminated number is more often right).
   Put that decision in "series_found", "figure_number_found" and
   "caption" — the caption text must be copied VERBATIM from the matching
   entry in candidates_found, not paraphrased, shortened, or corrected for
   apparent OCR errors.

If the text contains real captions for (series, number) combinations NOT in
the possible list, do not put them in candidates_found — they belong to a
different figure elsewhere in the document, not this one. Likewise do not
count a sentence that merely CITES or references a figure in passing as a
candidate — only an actual caption block, which is normally a longer,
self-contained description starting with one of the label forms above.

If candidates_found ends up empty, set "found" to false and leave
"series_found"/"figure_number_found"/"caption" null rather than guessing or
returning a citation sentence instead.

Return ONLY a JSON object, with fields in exactly this order:
{"candidates_found": [{"series": "main"|"extended_data", "figure_number": <int>, "page": <int>}, ...], "found": true|false, "series_found": "main"|"extended_data"|null, "figure_number_found": <int or null>, "caption": "<verbatim text or null>"}
"""


def parse_pages(text_docling_path: Path) -> dict[int, str]:
    text = text_docling_path.read_text(encoding="utf-8")
    parts = re.split(r'=== Page (\d+) ===', text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def parse_series_and_number(caption: str | None) -> tuple[str | None, int | None]:
    """Nature-family papers commonly run TWO independent figure-numbering
    sequences that both restart at 1: plain "Fig./Figure N" (main) and a
    separately-numbered "Extended Data Fig./Figure N" (extended_data). A
    bare number is ambiguous without knowing which sequence it belongs to
    — e.g. "Extended Data Figure 2" and main "Figure 2" are two different
    figures that happen to share a number. Check the more specific
    "Extended Data" form first since it's a superset match of the plain
    one. Returns (None, None) if the caption doesn't start with either
    recognizable form (multi-panel figures, uncaptioned plates, etc.)."""
    if not caption:
        return None, None
    text = caption.strip()
    m = re.match(r'^extended\s+data\s+fig(?:ure)?\.?\s*(\d+)', text, re.IGNORECASE)
    if m:
        return "extended_data", int(m.group(1))
    m = re.match(r'^fig(?:ure)?\.?\s*(\d+)', text, re.IGNORECASE)
    if m:
        return "main", int(m.group(1))
    return None, None


def build_candidate_pool(
    figures: list[dict], idx: int, confirmed: dict[str, tuple[str, int]]
) -> list[tuple[str, int]]:
    """(series, number) pairs figures[idx] could possibly be: not already
    claimed by anything else IN THAT SERIES anywhere in the document, AND
    within the range bounded by its nearest same-series confirmed
    neighbors in document order. The two series are eliminated completely
    independently — a number claimed in "main" says nothing about whether
    that same number is free in "extended_data", and vice versa.

    "extended_data" is only ever offered as a candidate series if the
    document already has at least one Docling-confirmed extended_data
    figure — most papers are single-series and have no such thing. Without
    this gate, an ordinary single-series document would still get handed
    open-ended "extended_data" candidates (nothing bounds an entirely
    absent series), which does nothing but invite the model to mislabel a
    perfectly ordinary "Fig. N" caption's series — seen in practice across
    several single-series documents in this corpus."""
    present_series = {s for (s, _) in confirmed.values()}
    candidates: list[tuple[str, int]] = []
    for series in ("main", "extended_data"):
        if series != "main" and series not in present_series:
            continue
        series_confirmed = {n for (s, n) in confirmed.values() if s == series}
        upper_bound = max(series_confirmed | {len(figures)})
        unclaimed = sorted(set(range(1, upper_bound + 1)) - series_confirmed)

        lo = hi = None
        for j in range(idx - 1, -1, -1):
            entry = confirmed.get(figures[j]["file"])
            if entry is not None and entry[0] == series:
                lo = entry[1]
                break
        for j in range(idx + 1, len(figures)):
            entry = confirmed.get(figures[j]["file"])
            if entry is not None and entry[0] == series:
                hi = entry[1]
                break

        candidates.extend(
            (series, n) for n in unclaimed
            if (lo is None or n > lo) and (hi is None or n < hi)
        )
    return candidates


def resolve_document(
    doc_dir: Path,
    model: str,
    base_url: str,
    temperature: float,
    timeout: float,
    window_before: int,
    window_after: int,
) -> dict:
    """Returns {"resolved": int, "attempted": int, "skipped_no_anchor": bool}."""
    fig_json_path = doc_dir / "figures" / "figures.json"
    text_path = doc_dir / "text_docling.txt"
    if not fig_json_path.is_file() or not text_path.is_file():
        return {"resolved": 0, "attempted": 0, "skipped_no_anchor": False}

    manifest = json.loads(fig_json_path.read_text(encoding="utf-8"))
    figures = manifest["figures"]
    pages = parse_pages(text_path)

    # Backfill series+number for every figure that already has a caption,
    # from the caption text itself rather than trusting a possibly-stale
    # printed_figure_number field — this also fixes any earlier figure
    # whose caption was a plain "Extended Data Fig. N" that an older,
    # non-series-aware parse left as printed_figure_number: None.
    confirmed: dict[str, tuple[str, int]] = {}
    for f in figures:
        series, number = parse_series_and_number(f.get("caption"))
        if series is None:
            continue
        confirmed[f["file"]] = (series, number)
        if f.get("printed_figure_number") != number or f.get("figure_series") != series:
            f["printed_figure_number"] = number
            f["figure_series"] = series

    unresolved = [f["file"] for f in figures if not f.get("caption")]
    if not unresolved:
        return {"resolved": 0, "attempted": 0, "skipped_no_anchor": False}
    if not confirmed:
        print(f"  [skip] {doc_dir.name}: no Docling-confirmed captions to anchor elimination against")
        return {"resolved": 0, "attempted": 0, "skipped_no_anchor": True}

    by_file = {f["file"]: i for i, f in enumerate(figures)}
    resolved = 0
    attempted = 0

    while unresolved:
        # Most-constrained-first: resolving the easiest figure tightens
        # everyone else's candidate pool before they're attempted.
        best_file, best_pool = None, None
        for file in unresolved:
            pool = build_candidate_pool(figures, by_file[file], confirmed)
            if not pool:
                continue
            if best_pool is None or len(pool) < len(best_pool):
                best_file, best_pool = file, pool

        if best_file is None:
            break  # every remaining figure has an empty candidate pool

        unresolved.remove(best_file)
        idx = by_file[best_file]
        fig = figures[idx]
        page = fig["page"]
        window_pages = [p for p in range(page - window_before, page + window_after + 1) if p in pages]
        window_text = "\n\n".join(f"--- Page {p} ---\n{pages[p]}" for p in window_pages)

        pool_str = ", ".join(f'{{"series": "{s}", "figure_number": {n}}}' for s, n in best_pool)
        attempted += 1
        result = extract_json(
            system_prompt=SYSTEM_PROMPT,
            user_content=(
                f"POSSIBLE (series, number) CANDIDATES (must be one of these, no other combination is valid for this figure): [{pool_str}]\n\n"
                f"THIS FIGURE'S OWN PAGE: {page} (use this to judge which candidate caption, if more than one is present, is the best match)\n\n"
                f"OCR TEXT (pages {window_pages[0]}-{window_pages[-1]}):\n{window_text}"
            ),
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            label=f"{doc_dir.name}:{best_file}",
        )

        if not isinstance(result, dict) or not result.get("found"):
            continue

        found_series = result.get("series_found")
        found_n = result.get("figure_number_found")
        if (found_series, found_n) not in best_pool:
            print(f"    [warn] {best_file}: LLM found {found_series} Fig. {found_n}, "
                  f"not in candidate pool {best_pool} — discarded")
            continue

        caption = result.get("caption")
        if not caption:
            continue

        # Cross-check the model's self-reported (series, number) against
        # what the returned caption text itself actually starts with — pool
        # membership alone isn't enough. Seen in practice: a model can
        # return a real caption VERBATIM while misreporting which number it
        # belongs to (e.g. returning "Figure 3 | Chronostratigraphic..."
        # tagged as figure_number_found: 4). Since the caption is quoted
        # verbatim, its own printed number is a hard, cheap, deterministic
        # check independent of anything the model claims about itself.
        parsed_series, parsed_n = parse_series_and_number(caption)
        if (parsed_series, parsed_n) != (found_series, found_n):
            print(f"    [warn] {best_file}: LLM claimed {found_series} Fig. {found_n}, but the "
                  f"returned caption text itself starts with {parsed_series} Fig. {parsed_n} — discarded")
            continue

        fig["caption"] = caption
        fig["printed_figure_number"] = found_n
        fig["figure_series"] = found_series
        confirmed[best_file] = (found_series, found_n)
        resolved += 1
        label_prefix = "Extended Data Fig" if found_series == "extended_data" else "Fig"
        print(f"    [recovered] {best_file}: {label_prefix} {found_n} — {caption[:80]}...")

        # Re-stamp the watermark with the now-confirmed real number (the
        # filename itself is left alone — renaming would orphan any
        # already-produced downstream output, e.g. site_form_segmenter's
        # classify_figures.py run, that references the original filename)
        # and append the caption box, same finishing step _extract_figures
        # already applies at extraction time.
        img_path = doc_dir / "figures" / best_file
        if img_path.is_file():
            from PIL import Image
            image = Image.open(img_path)
            image = stamp_figure_label(image, f"{label_prefix} {found_n} - p{page}")
            image = compose_figure_with_caption(image, caption)
            _save_image_with_retry(image, img_path)

    fig_json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"resolved": resolved, "attempted": attempted, "skipped_no_anchor": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_fill_captions.yaml")
    parser.add_argument("--stem", default=None, help="Process only this document folder (by name)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    model = cfg.get("model", "qwq:32b")
    base_url = cfg.get("base_url", "http://localhost:11434")
    temperature = cfg.get("temperature", 0.05)
    timeout = cfg.get("timeout_seconds", 900)
    window_before = cfg.get("window_pages_before", 1)
    window_after = cfg.get("window_pages_after", 2)

    doc_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and (d / "figures" / "figures.json").is_file())
    if args.stem:
        doc_dirs = [d for d in doc_dirs if d.name == args.stem]
        if not doc_dirs:
            print(f"[error] no document folder named '{args.stem}' under {output_dir}")
            return

    print(f"Model    : {model}")
    print(f"Documents: {len(doc_dirs)}")

    total_resolved = total_attempted = total_skipped_docs = 0
    for doc_dir in doc_dirs:
        print(f"\n[{doc_dir.name}]")
        stats = resolve_document(doc_dir, model, base_url, temperature, timeout, window_before, window_after)
        total_resolved += stats["resolved"]
        total_attempted += stats["attempted"]
        total_skipped_docs += int(stats["skipped_no_anchor"])
        if stats["attempted"]:
            print(f"  {stats['resolved']}/{stats['attempted']} recovered")

    print(f"\nTotal recovered: {total_resolved}/{total_attempted} attempted "
          f"({total_skipped_docs} document(s) skipped — no confirmed anchor)")


if __name__ == "__main__":
    main()
