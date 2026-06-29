# pdf_ocr

This pipeline converts a corpus of scanned or born-digital PDFs into structured, machine-readable outputs for downstream LLM use, including RAG indexing, agent-crawlable wikis, and full-text search. Each PDF is processed in two complementary passes: Surya OCR reads the raw page imagery (handling both printed text and handwriting), and Docling uses that recognized text alongside its own layout detection to recover document structure, including section headers, tables, and reading order, and exports clean Markdown. The result per document is a Markdown file suitable for chunking and embedding, a searchable PDF with an invisible text layer, and supporting JSON and plain-text outputs.

Every tool in this ecosystem takes pdf_ocr output as its starting input. The two downstream tools built directly on top of pdf_ocr output are [site_form_segmenter](https://github.com/jnpaige/site_form_segmenter), which maps out the structure of each document, and [text_coding_program](https://github.com/jnpaige/text_coding_program), a browser-based interface for human coding of the same documents.

---

## Output

For each input PDF a subfolder is created containing four files. The Markdown file (`<pdf_stem>.md`) preserves section headers, tables, and reading order and is the primary output for RAG indexing and LLM agent ingestion. The searchable PDF (`<pdf_stem>_ocr.pdf`) is the original PDF with an invisible Surya text layer added, making it fully text-searchable and selectable; this file is omitted when `docling.do_ocr` is set to false. The structured JSON file (`ocr_docling.json`) contains per-page results with text blocks and confidence scores. The plain text file (`text_docling.txt`) is a dump of each page separated by `=== Page N ===` markers, with heading items prefixed by markdown hash markers matching the `.md` output.

Both `text_docling.txt` and `<pdf_stem>.md` are kept. The txt file is the primary input for downstream LLM tools because it carries both page-indexed boundaries and heading structure in one place. The md file is retained as a clean reading copy and for RAG and vector-store ingestion.

The downstream tools each consume different output files. The `text_docling.txt` file feeds into [site_form_segmenter](https://github.com/jnpaige/site_form_segmenter) for page classification, and into site_coder, site_attribute_extractor, and site_vocab_extractor for LLM prompt text. The `.md` file feeds into site_wiki for extraction and synthesis. The `_ocr.pdf` feeds into [text_coding_program](https://github.com/jnpaige/text_coding_program) for the PDF page viewer, and into site_form_segmenter in vision mode for contact sheet rendering.

---

## Downstream workflow 1: Site forms

Site forms are short documents, typically 2–15 pages, each filed under one trinomial. The trinomial is often somewhere in the source filename. Each form may contain multiple investigations appended together, and interpretations and findings differ between those investigations. Gathering standardized data about each investigation across a large corpus of these PDFs requires isolating which pages relate to each investigation, and then routing only those pages to whatever LLM or human coder is doing the extraction. That page-scoping also makes LLM performance more predictable by avoiding the noise of sending irrelevant pages alongside the information you want to extract.

The [site_form_segmenter](https://github.com/jnpaige/site_form_segmenter) handles this first. It runs a 4-pass LLM approach over each document: identifying investigation boundaries, then classifying each investigation's pages into form pages, narrative pages, and NRHP eligibility pages. The output is a structured page map stored as a `.segments.json` file for each site. All downstream tools read that map and limit their inputs to the pages it specifies.

Codebook-driven prompting is central to how extraction is designed here. site_coder uses the parsed codebook entries directly as LLM prompts, so LLMs receive instructions analogous to what you would give human coders. That enables a direct comparison with archaeologists or other expert coders on the same coding tasks. The [text_coding_program](https://github.com/jnpaige/text_coding_program) produces output in the same JSON schema as site_coder, enabling inter-rater reliability analysis among people and models working from the same documents and the same codebook.

A single site form may contain 1–5 investigations appended chronologically. A 12-page form with 3 investigations might result in only 2–4 pages being sent per LLM call after page-scoping, which helps keep smaller local LLMs viable — a key goal for an open science approach to this kind of analysis.

The downstream consumers after segmentation are site_coder (LLM coding against a structured codebook), site_attribute_extractor (focused extraction prompts designed for LLM rather than human coders), site_wiki (extraction and prose synthesis into an Obsidian-compatible wiki), and [text_coding_program](https://github.com/jnpaige/text_coding_program) (browser-based human coding interface). All share the same inputs: OCR text, the segment map, and a codebook.

---

## Downstream workflow 2: Archaeological reports

Reports are longer documents, typically 50–400+ pages, that discuss multiple sites. Unlike site forms, the trinomial is not the filename — one report may cover anywhere from one to hundreds of sites. Pulling standardized information about each site from a large report corpus requires an additional discovery step and a two-stage segmentation before any extraction can happen.

Passes 1 and 2 of this workflow can run in parallel since they both read the full report independently. The [site_form_segmenter](https://github.com/jnpaige/site_form_segmenter) runs first on the whole document to classify pages into structural sections — executive summary, methods, results, and recommendations. This pass benefits from a larger model (32b+) due to document length. Simultaneously, site_vocab_extractor scans the full report text to identify all trinomials and alternative site numbers mentioned. Its output includes a `source_file` column that links each discovered trinomial back to its source report, preventing cross-contamination when you later feed those trinomial lists to the second segmentation pass.

The second segmentation pass then iterates per trinomial: for each site the vocab extractor discovered, it searches the relevant section pages (from pass 1) for mentions of that specific site. The result is a per-trinomial page map scoped to just the report sections and pages that discuss that site. Feeding all trinomials to a single prompt might work for short reports with few sites, but archaeological reports vary enormously in length and site count, making that approach unpredictable at scale. Per-trinomial iteration isolates errors and keeps context focused at the cost of more LLM calls, but each call is short and predictably sized.

Once per-trinomial page maps exist, the same downstream tools as in the site form workflow handle extraction: site_coder, site_attribute_extractor, site_wiki, and [text_coding_program](https://github.com/jnpaige/text_coding_program).

---

## Installation

I use uv for Python environment management. It is far faster than conda or pip for snapping a repo to the right environment, it works identically on Windows, Mac, and Linux, and it never requires manually activating a virtual environment.

```powershell
# Windows — install uv once per machine
winget install astral-sh.uv

# Then in the repo directory (creates .venv and installs deps in seconds):
uv sync

# Run the pipeline
uv run python run.py
```

```bash
# Mac/Linux — install uv once per machine
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then in the repo directory:
uv sync
uv run python run.py
```

`uv.lock` is committed to the repo, so every machine gets identical package versions. If you prefer to activate the venv manually, run `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Mac/Linux, then use `python run.py` directly.

For GPU acceleration, which is recommended for large corpora:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

Set `pdf_input` and `output_dir` in `config.yaml`, then run:

```powershell
uv run python run.py

# or with a custom config:
uv run python run.py --config path/to/config.yaml
```

## Configuration

```yaml
pdf_input:  'C:\path\to\pdfs'     # single PDF or directory of PDFs
output_dir: 'C:\path\to\output'   # subfolders created per PDF

docling:
  # markdown_dir: 'C:\path\to\md' # optional: all .md files mirrored here (flat)
  # do_ocr: false                 # optional: skip Surya OCR (see below)
```

When `markdown_dir` is set, every `.md` file is also written to that directory with a flat layout — no subfolders. This is useful for pointing a vector store or RAG indexer at a single location. Set `do_ocr: false` when the input PDFs already have an embedded text layer, such as `_ocr.pdf` outputs from a previous run of this pipeline. Docling parses the existing text layer instead of running Surya, and the `_ocr.pdf` is not re-written.

## Dependencies

- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF reading and invisible text layer construction
- [Docling](https://github.com/DS4SD/docling) — layout detection, table extraction, Markdown export
- [Surya OCR](https://github.com/datalab-to/surya) — OCR backend (text detection + recognition, including handwriting)
- [PyYAML](https://pyyaml.org/) — config parsing
- [Transformers](https://huggingface.co/docs/transformers/) — required by Surya models
