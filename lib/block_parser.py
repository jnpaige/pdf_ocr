"""Split a document's page-indexed markdown text into addressable blocks —
paragraphs, headings, and individual table rows — for LLM tagging.

This exists because tagging needs stable, referenceable units: the LLM tags
a block by id, rather than free-quoting text back (which doesn't scale past
a handful of items and can't be reliably matched back to its source). Table
rows are split individually (not left as one block per table) because
comparison tables are exactly where cross-context evidence lives — a row
like "Cores | 12 | 8 | 3" needs its own tag set, not the whole table's.

Two corpus-wide OCR artifacts are normalised here rather than left for each
downstream pass to rediscover independently:

- Docling sometimes escapes inline HTML into heading text, so a heading
  arrives as "## &lt;b&gt;Koimilot Locus 2&lt;/b&gt;". Heading text is the
  primary evidence both for section classification and for cluster labels,
  so it is unescaped and de-tagged at parse time.

- A paragraph interrupted by a floated table or a page break resumes in a
  later block that starts mid-sentence ("parable to lavas that crop out
  ~15-20 km upstream ..."). The fragment has lost the subject it depends
  on and is effectively untaggable alone. `stitch_continuations` reattaches
  it to the prose block it continues. Note that the interrupted block does
  NOT reliably end mid-sentence — Docling frequently appends the floated
  table's footnote to it ("... Totals are revised from Tryon (2003).") — so
  detection keys on the fragment's own opening, never on how the previous
  block ends.
"""
import html
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r'^#{1,6}\s+')
_TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
_TABLE_SEP_RE = re.compile(r'^\s*\|[\s:|-]+\|\s*$')  # |---|---| separator row

_HTML_TAG_RE = re.compile(r'<[^>]+>')
# A block that is nothing but a Docling placeholder comment, e.g. "<!-- image -->".
_HTML_COMMENT_RE = re.compile(r'^<!--.*-->$')
# Table/figure captions, by the near-universal journal convention of opening
# with "Table 3.", "Fig. 2", "Figure 10". Used to skip over captions when
# looking backwards for a continuation's antecedent: a floated table's caption
# sits between the interrupted paragraph and the point where it resumes.
_CAPTION_RE = re.compile(r'^(?:table|fig(?:ure)?)\.?\s*\d+', re.IGNORECASE)
# A paragraph opening with a lowercase letter is the continuation signal.
# Body paragraphs in published papers effectively never start this way.
_CONTINUATION_RE = re.compile(r'^[a-z]')
# A word broken across the interruption ("com-" + "parable") rejoins with no space.
_SOFT_HYPHEN_RE = re.compile(r'[A-Za-z]-$')

# An antecedent must be substantial enough to be running prose before a
# fragment is merged into it. Length alone does the work here: it rejects the
# short lines a fragment might otherwise be glued to — Docling image labels
# ("Full page image", "Engineering drawing"), running heads, affiliations —
# without naming any of them, so it generalises across journals.
#
# Deliberately NOT also requiring the antecedent to contain sentence-ending
# punctuation: a paragraph interrupted by a floated table before its first
# full stop contains none, and that is precisely the case this exists to fix.
MIN_ANTECEDENT_WORDS = 25

# A continuation must itself be long enough to be resumed prose. Without this,
# a stray lowercase figure-panel label ("a", "b" — Docling emits these as their
# own blocks next to plate figures) gets glued onto the end of whatever prose
# precedes it. A genuine resumed sentence is many words long; nothing of value
# is lost by declining the very short ones.
MIN_FRAGMENT_WORDS = 3
# How far a continuation may reach back for its antecedent. 1 covers the
# common "paragraph resumes after a table that straddles a page break" case
# while refusing to reach back an arbitrary distance.
MAX_PAGE_SPILL = 1


@dataclass
class Block:
    block_id: str
    page: int
    type: str  # "heading" | "paragraph" | "table_row"
    text: str


def _clean_heading(text: str) -> str:
    """Unescape entity-encoded markup and drop inline tags from heading text."""
    text = _HTML_TAG_RE.sub('', html.unescape(text))
    return re.sub(r'\s+', ' ', text).strip()


def is_placeholder(text: str) -> bool:
    """True for a Docling placeholder comment — no content, only noise."""
    return bool(_HTML_COMMENT_RE.match(text.strip()))


def is_caption(text: str) -> bool:
    return bool(_CAPTION_RE.match(text.strip()))


def _is_prose(text: str) -> bool:
    """Substantial running prose — a safe thing to append a fragment to."""
    return len(text.split()) >= MIN_ANTECEDENT_WORDS


def _split_page(page_num: int, page_text: str, counter: list[int]) -> list[Block]:
    """counter is a single-element mutable list used as a running block index
    within the page, so table-row handling (which consumes multiple raw
    lines per emitted block) can share the same numbering as paragraphs."""
    blocks: list[Block] = []
    lines = page_text.splitlines()
    i = 0
    para_buf: list[str] = []

    def flush_para():
        if para_buf:
            text = " ".join(l.strip() for l in para_buf if l.strip())
            if text:
                counter[0] += 1
                blocks.append(Block(f"p{page_num}_b{counter[0]}", page_num, "paragraph", text))
            para_buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        if _HEADING_RE.match(stripped):
            flush_para()
            counter[0] += 1
            blocks.append(Block(
                f"p{page_num}_b{counter[0]}", page_num, "heading", _clean_heading(stripped),
            ))
            i += 1
            continue

        if _TABLE_ROW_RE.match(stripped):
            flush_para()
            # Collect the contiguous run of table lines.
            table_lines = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            if not table_lines:
                continue
            header = table_lines[0]
            data_rows = [
                l for l in table_lines[1:] if not _TABLE_SEP_RE.match(l)
            ]
            for row in data_rows:
                counter[0] += 1
                blocks.append(Block(
                    f"p{page_num}_b{counter[0]}", page_num, "table_row",
                    f"columns: {header}  ||  row: {row}",
                ))
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    return blocks


def split_blocks(pages: dict[int, str]) -> list[Block]:
    """Split every page's markdown text into addressable blocks, in page
    order. Block ids are unique within the document (p{page}_b{n})."""
    all_blocks: list[Block] = []
    for page_num in sorted(pages):
        counter = [0]
        all_blocks.extend(_split_page(page_num, pages[page_num], counter))
    return all_blocks


def _find_antecedent(kept: list[Block], fragment: Block) -> int | None:
    """Index in `kept` of the prose block `fragment` continues, or None.

    Walks backwards past the things that legitimately interrupt a paragraph
    — a floated table's rows, its caption, an image placeholder — and stops
    at anything that makes the antecedent unknowable: a heading (a new
    section started), too many pages back, or a neighbouring paragraph too
    slight to be running prose. None is the safe outcome: the fragment
    stays its own block, exactly as before this function existed.
    """
    for idx in range(len(kept) - 1, -1, -1):
        blk = kept[idx]
        if blk.page < fragment.page - MAX_PAGE_SPILL:
            return None
        if blk.type == "heading":
            return None
        if blk.type == "table_row":
            continue
        if is_placeholder(blk.text) or is_caption(blk.text):
            continue
        return idx if _is_prose(blk.text) else None
    return None


def _join(antecedent: str, fragment: str) -> str:
    if _SOFT_HYPHEN_RE.search(antecedent):
        return antecedent[:-1] + fragment
    return f"{antecedent} {fragment}"


def stitch_continuations(blocks: list[Block]) -> tuple[list[Block], int]:
    """Reattach mid-sentence paragraph fragments to the prose they continue.

    Returns (blocks, n_merged). A merged block keeps the antecedent's id and
    page, so a fragment stranded on the far side of a page break is scoped
    to the page its sentence started on.
    """
    kept: list[Block] = []
    merged = 0
    for b in blocks:
        if (
            b.type == "paragraph"
            and _CONTINUATION_RE.match(b.text)
            and len(b.text.split()) >= MIN_FRAGMENT_WORDS
            and not is_caption(b.text)
        ):
            idx = _find_antecedent(kept, b)
            if idx is not None:
                ant = kept[idx]
                kept[idx] = Block(ant.block_id, ant.page, ant.type, _join(ant.text, b.text))
                merged += 1
                continue
        kept.append(b)
    return kept, merged


def format_blocks_for_prompt(blocks: list[Block]) -> str:
    return "\n".join(f"[{b.block_id}] {b.text}" for b in blocks)
