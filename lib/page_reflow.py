"""Geometric reflow of a page whose layout docling failed to recover.

When Docling's layout model cannot segment a page, the pipeline falls back to
Surya's raw OCR cells. Those cells are accurate but were ordered by a simple
15pt horizontal band sort, which on a two-column page alternates between the
columns line by line, so the resulting text reads as two interleaved articles.
This module recovers the reading order from the line geometry alone.

Measured over the 77 fallback pages in the procedural-unit corpora: column
switches in the reading order fall from 1,825 to 165, and 5,379 separate lines
become about 800 paragraphs with 336 hyphenated words rejoined.

Works only from the line bounding boxes already stored in ocr_docling.json.
All coordinates are BOTTOMLEFT origin, so a larger `t` means higher on the
page and reading top-to-bottom means sorting by descending `t`.
"""
import statistics
from collections import Counter


def find_gutters(lines, bin_pt=2.0, cross_frac=0.03, min_w_frac=0.012):
    """Vertical bands of the page that almost no line of text crosses.

    Each such band is the white space between two columns. The tolerances
    matter: bins must be small (a 6pt bin smears a 10pt gutter away), the
    minimum width must be low (real gutters in this corpus are 6-10pt on a
    490pt text width, i.e. about 1.5%), and a few lines are allowed to cross
    because page furniture such as a JSTOR footer genuinely spans columns.
    """
    x0 = min(t["bbox"]["l"] for t in lines)
    x1 = max(t["bbox"]["r"] for t in lines)
    width = x1 - x0
    if width <= 0:
        return [], x0, x1
    n_bins = int(width / bin_pt) + 1
    coverage = [0] * n_bins
    for line in lines:
        box = line["bbox"]
        first = max(0, int((box["l"] - x0) / bin_pt))
        last = min(n_bins - 1, int((box["r"] - x0) / bin_pt))
        for i in range(first, last + 1):
            coverage[i] += 1

    threshold = max(1, int(cross_frac * len(lines)))
    runs, start = [], None
    for i, count in enumerate(coverage):
        if count <= threshold and start is None:
            start = i
        elif count > threshold and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n_bins - 1))

    gutters = []
    for first, last in runs:
        gx0, gx1 = x0 + first * bin_pt, x0 + (last + 1) * bin_pt
        wide_enough = (gx1 - gx0) >= min_w_frac * width
        not_a_margin = gx0 > x0 + 0.05 * width and gx1 < x1 - 0.05 * width
        if wide_enough and not_a_margin:
            gutters.append((gx0, gx1))
    return gutters, x0, x1


def column_of(line, gutters, x0, x1):
    """Which column a line sits in, or None if it spans a gutter."""
    box = line["bbox"]
    for gx0, gx1 in gutters:
        if box["l"] < gx0 - 2 and box["r"] > gx1 + 2:
            return None
    bounds = [x0] + [g[0] for g in gutters] + [x1]
    centre = (box["l"] + box["r"]) / 2
    for i in range(len(bounds) - 1):
        if bounds[i] <= centre <= bounds[i + 1]:
            return i
    return 0



def _split_into_paragraphs(column_lines):
    """Group one column's lines, already ordered top to bottom, into paragraphs.

    Splits on PITCH -- the distance between consecutive line tops -- rather
    than on the gap between box edges. Two reasons, both measured on this
    corpus: Surya's line boxes routinely overlap vertically (7-9 negative
    edge-gaps per column on Chazan and Desrosiers), which makes edge-gap
    arithmetic unreliable; and pitch separates the two populations cleanly
    while edge-gap does not. Body lines sit at 10-13pt of pitch and paragraph
    breaks at 20-222pt, with nothing in between, so any threshold from 1.4x
    to 1.8x the median gives an identical answer.

    An earlier version also split on first-line indentation and on the
    previous line stopping short of the right margin. Both are dropped: the
    indent rule fired on every line of a column whose body text is outdented
    relative to its first line, and the short-line rule is meaningless in
    ragged-right text, which most of these pages are.
    """
    if not column_lines:
        return [], 0

    pitches = [a["bbox"]["t"] - b["bbox"]["t"]
               for a, b in zip(column_lines, column_lines[1:])]
    positive = [p for p in pitches if p > 0]
    typical_pitch = statistics.median(positive) if positive else 0

    groups, current = [], [column_lines[0]]
    for i, line in enumerate(column_lines[1:]):
        starts_new_paragraph = typical_pitch and pitches[i] > 1.5 * typical_pitch
        if starts_new_paragraph:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    groups.append(current)

    paragraphs, hyphen_joins = [], 0
    for group in groups:
        text = ""
        for line in group:
            piece = line["text"].strip()
            if not text:
                text = piece
            elif text.endswith("-") and text[-2:-1].isalpha():
                text = text[:-1] + piece      # "bur-" + "ins" -> "burins"
                hyphen_joins += 1
            else:
                text += " " + piece
        paragraphs.append({"text": text, "lines": group})
    return paragraphs, hyphen_joins


def reflow(lines):
    """Return (paragraphs, gutters, ordered_lines, hyphen_joins) for one page.

    Columns are detected once for the page. A line that spans a gutter -- a
    running head, a full-width heading, a wide caption -- closes the current
    band, so reading order becomes heading, then column one, then column two,
    rather than alternating between the columns.
    """
    gutters, x0, x1 = find_gutters(lines)
    top_to_bottom = sorted(lines, key=lambda t: -t["bbox"]["t"])

    bands, band = [], {}
    for line in top_to_bottom:
        column = column_of(line, gutters, x0, x1)
        if column is None:
            if band:
                bands.append(band)
                band = {}
            bands.append({"full_width": [line]})
        else:
            band.setdefault(column, []).append(line)
    if band:
        bands.append(band)

    paragraphs, ordered, joins = [], [], 0
    for band in bands:
        if "full_width" in band:
            for line in band["full_width"]:
                paragraphs.append({"text": line["text"].strip(), "lines": [line]})
                ordered.append(line)
            continue
        for column in sorted(band):
            column_paragraphs, n = _split_into_paragraphs(band[column])
            paragraphs += column_paragraphs
            ordered += band[column]
            joins += n
    return paragraphs, gutters, ordered, joins


def count_column_switches(ordered_lines, gutters, x0, x1):
    """How many times the reading order jumps from one column to another.

    On a correctly ordered two-column page this is small: one switch per
    band. On the current band-sorted order it is large, because the two
    columns alternate line by line.
    """
    switches = 0
    previous = None
    for line in ordered_lines:
        column = column_of(line, gutters, x0, x1)
        if column is not None and previous is not None and column != previous:
            switches += 1
        if column is not None:
            previous = column
    return switches
