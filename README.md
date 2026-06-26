# pdf_ocr

This pipeline converts a corpus of scanned or born-digital PDFs into structured, machine-readable outputs for downstream LLM use, including RAG indexing, agent-crawlable wikis, and full-text search. Each PDF is processed in two complementary passes: Surya OCR reads the raw page imagery (handling both printed text and handwriting), and Docling uses that recognized text alongside its own layout detection to recover document structur,e which includes section headers, tables. The result per document is a Markdown file suitable for chunking and embedding, a searchable PDF with an invisible text layer, and supporting JSON and plain-text outputs.

Every tool in the text extraction ecosystem developed here, segmenter, coder, extractor, wiki generator, human coding program, takes pdf_ocr output as its starting input. 

## Output

For each input PDF a subfolder is created:

```
<output_dir>/<pdf_stem>/
  <pdf_stem>.md        Markdown — section headers, tables, and reading order preserved;
                       primary output for RAG indexing and LLM agent ingestion
  <pdf_stem>_ocr.pdf   Original PDF with an invisible Surya text layer added;
                       fully text-searchable and selectable
                       (omitted when docling.do_ocr is false)
  ocr_docling.json     Structured per-page results (text blocks with confidence scores)
  text_docling.txt     Plain text dump — one "=== Page N ===" section per page,
                       with heading items prefixed by markdown # markers
                       (# Title, ## Section, ### Subsection, matching .md output)
```

Both `text_docling.txt` and `<pdf_stem>.md` are kept. The txt file is the primary input for downstream LLM tools because it carries both page-indexed boundaries and heading structure in one place. The md file is retained as a clean reading copy and for RAG/vector-store ingestion.

**Note:** Earlier version had a bug where docling's markdown exporter (`export_to_markdown`) and the custom `_build_page_results()` function produced complementary but separate outputs: the `.md` had heading markers but no page boundaries, while `text_docling.txt` had page boundaries but no heading markers. Pipelines that needed both had to cross-reference the two files (e.g. search each heading string from the .md against txt pages to determine its docling page index). The current version embeds heading markup directly in `text_docling.txt`, eliminating that merge step.

Downstream tools consume different outputs:

| File | Consumed by |
|---|---|
| `text_docling.txt` | site_form_segmenter (page classification), site_coder (LLM prompt text), site_attribute_extractor, site_vocab_extractor |
| `<pdf_stem>.md` | site_wiki (extraction), RAG/vector-store indexing |
| `<pdf_stem>_ocr.pdf` | text_coding_program (PDF page viewer for human coding), site_form_segmenter (vision mode contact sheet) |

---

## Downstream workflow 1: Site forms

Site forms are short documents (2–15 pages), each filed under one trinomial. The trinomial is often somewhere in the source filename. Each form may contain multiple investigations appended together, and interpretations and findings differ between those investigations. We may want to gather data about each investigations findings across these pdfs, which requires isolating pages that relate to each investigation. We may also want to make the model performance more predictable by not bombarding it with all the information in the site form but instead focus only on relevant pages to he information we want to extract. 


### Design decisions for site forms

A single site form may contain 1–5 investigations appended chronologically. The segmenter identifies where each investigation starts and which pages belong to it. A page map reduces amount of information passed on to LLMs. Instead of sending the entire form to the LLM, each tool sends only the pages relevant to the current investigation and page type. A 12-page form with 3 investigations might send 2–4 pages per LLM call, which helps us incorporate smaller local LLMs into the worflow - A key goal for an open science approach to this kind of thing. 
Codebook-driven prompting is another goal to help support an open science approach to using LLMs to pull data from text. Site_coder uses the parsed codebook entries directly as LLM prompts. So, LLMs get instructions analogous to what you would give human coders. That enables a comparison with archaeologists or other expert coders on the same tasks. Text_coding_program produces output in the same JSON schema as site_coder, enabling direct inter-rater reliability analysis on coding tasks with reference to the same information, and same codebook, among people and transformers. 

Below is a schematic for how this kind of problem is addressed using one workflow that builds off of the pdf_ocr output. 

```
pdf_ocr
  │
  │  Per-site output: <trinomial>/text_docling.txt + _ocr.pdf + .md
  │
  ├──► site_form_segmenter
  │      Segments each form into investigations with page-type assignments.
  │      4-pass LLM approach: investigation boundaries → form pages →
  │      narrative pages → NRHP pages.
  │      Output: <trinomial>.segments.json with investigation boundaries
  │      and typed page lists (form_pages, narrative_pages, nrhp_pages).
  │
  │      Page types and keys are ingested by scripts downstream.
  │      Any *_pages key in the segment JSON is recognized as a scope.
  │
  ├──► codebook_tools
  │      Parses .Rmd codebooks into per-trait JSON files (full_text + data_type).
  │      Not part of this pipeline, but its output drives site_coder.
  │
  ▼
  Three downstream consumers share the same inputs
  (OCR text + segment map + codebook):
  │
  ├──► site_coder
  │      This tasks an LLM with coding text using a codebook designed for archaeologists. For each investigation and for each trait,
  │      pull in the the relevant pages (scoped by segment map),and then construct
  │      a prompt from the relevant codebook entrie, and send the prompt to a model through Ollama.
  │      Output: per-site .coded.json and a corpus-wide all_coded.xlsx which merges the info from the json files.
  │      
  │
  ├──► site_attribute_extractor
  │      This applies a prompt to the text coding problem that isn't meant to be used by archaeologists and is designed for LLMs
  │      Each extractor is one focused LLM
  │      call per investigation, using only the relevant page scope.
  │      Output: per-site .attributes.json + corpus-wide .xlsx.
  │
  ├──► site_wiki
  │      Generates a local wikipedia based on documents. LLM extraction (structured JSON per report) is performed before a separate synthesis call
  │      which produces a prose wiki page per trinomial. Produces
  │      Obsidian-compatible Markdown with frontmatter, tags, and
  │      wikilinks. Supports multi-model extraction/synthesis comparison.
  │
  └──► text_coding_program
         Browser-based interactive coding program for archaeologists. Renders _ocr.pdf pages,
         displays codebook entries, saves .coded.json in the same
         format as site_coder for direct IRR comparison.
```


---

## Downstream workflow 2: Archaeological reports

Reports are longer documents (50–400+ pages) that discuss multiple sites. The trinomial is *not* the filename — one report may report one or many sites. One way to tackle this is through adding a trinomial extraction step for that document, and running segmentation in two stages, one to isolate relevant sections of the report, and one isolating pages in those sections relevant to each trinomial mentioned in the report. That subset of pages is then passed onto LLM coder/extractors who focus just on coding information about one site. 

### Design decisions for reports

Unlike in the site forms, where the filename contains the trinomial, or where one pdf relates to a single site, or two related sites, longer reports can contain discussion of one, or hundreds of sites. So, if the goal is to pull a standardized set of info about each site, from each report, that influences how the outputs of pdf_ocr are treated. Here, we need to run the vocab extractor to identify all site names and trinomials in each report, and also narrow down the relevant sections of report, while also attenuating the LLM calls so their context is not overwhelmed and the inputs are relatively predictably sized. 
The vocab extractor's `source_file` column links each discovered trinomial back to its source report, preventing cross-contamination when feeding trinomial lists to pass 2 (we dont want to iterate across trinomials that aren't in a given report). 

For now we approach this with a two-stage segmentation. Pass 1 identifies the report's structural sections. Pass 2 identifies per-trinomial pages within those sections (iterates per trinomial, scoped to the section pages from pass 1). The resulting map, pages from relevant sections that discuss the trinomial in question, should help keep each LLM call manageable.

If we were to Feed all trinomials to a single prompt that forces the model to hold the full text and full trinomial list and produce structured output for each, this would be ok if there are small site counts, and short reports. But that is unpredictable across site reports in archaeology. Per-trinomial iteration isolates errors and keeps context focused, and should produce more predictable outputs, at the cost of expanding the number of runs. 

The whole-document structural pass benefits from 32b+ models. Per-trinomial pass 2 could work well with 14b models since the text scope is pre-filtered.


```
pdf_ocr
  │
  │  Per-report output: <report_name>/text_docling.txt + _ocr.pdf + .md
  │
  ├──► site_form_segmenter (pass 1 — report sections)
  │      Segments the full report into structural sections:
  │      executive summary, methods, results, recommendations.
  │      Output: section-level page map.
  │
  │      This pass operates on the whole document and benefits
  │      from a larger model (32b+) due to the document length.
  │
  ├──► site_vocab_extractor (trinomial discovery)
  │      Scans the full report text to extract all trinomials mentioned.
  │      Uses a configurable prompt (prompt_file: in config.yaml).
  │      Output: per-report .terms.json with trinomial list + source_file
  │      column in the aggregate CSV, linking trinomials to their
  │      source report.
  │
  │      This step tells us WHICH sites the report covers.
  │
  ├──► site_form_segmenter (pass 2 — per-trinomial pages)
  │      For each discovered trinomial, identifies which pages within
  │      each structural section discuss that specific site.
  │      Input: section page map (from pass 1) + trinomial list
  │      (from vocab extractor).
  │      Output: per-trinomial segment map with section-scoped pages.
  │
  │      Iterating per trinomial keeps context manageable — a 300-page
  │      report's results section might be 40 pages, and searching
  │      40 pages for one trinomial is easy for even a 14b model. For now we avoid doing multiple trinomials in a call because things can balloon.
  │
  ▼
  Same downstream consumers as site forms:
  site_coder, site_attribute_extractor, site_wiki, text_coding_program
```



---

## Installation

I use UV for python installations now. It is way easier than conda/pip, and setting up virtual environments on each machine.
Also, if you are using this to process lots of pdfs you may want to run it on multiple machines. UV is a package manager that runs on rust
and is far and away faster than any other option I've found for snapping a repo to the right computational environment. 

```powershell
# Windows — install uv once per machine
winget install astral-sh.uv

# Then in the repo directory (creates .venv and installs deps in seconds):
uv sync

# Run the pipeline
uv run python run.py
# or activate the venv and use python directly:
.venv\Scripts\activate
python run.py
```

```bash
# Mac/Linux — install uv once per machine
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then in the repo directory:
uv sync
uv run python run.py
```

`uv.lock` is committed to the repo, so every machine gets identical package versions.

**For GPU acceleration (recommended for large corpora):**

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Alternative — plain pip:**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# GPU torch override:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

1. Set `pdf_input` and `output_dir` in `config.yaml`
2. Run:

```powershell
uv run python run.py

# or with a custom config:
uv run python run.py --config path/to/config.yaml
```

## Configuration

```yaml
# config.yaml

pdf_input:  'C:\path\to\pdfs'     # single PDF or directory of PDFs
output_dir: 'C:\path\to\output'   # subfolders created per PDF

docling:
  # markdown_dir: 'C:\path\to\md' # optional: all .md files mirrored here (flat)
  # do_ocr: false                 # optional: skip Surya OCR (see below)
```

### `markdown_dir`

When set, every `.md` file is also written to this directory with a flat layout — no subfolders. Useful for pointing a vector store or RAG indexer at a single location:

```yaml
docling:
  markdown_dir: 'C:\corpus\markdown'
```

### `do_ocr`

Set to `false` when the input PDFs already have an embedded text layer (e.g. they
are `<pdf_stem>_ocr.pdf` outputs from a previous run of this pipeline). Docling
parses the existing text layer instead of running Surya, and the redundant
`<pdf_stem>_ocr.pdf` is not written:

```yaml
docling:
  do_ocr: false
```

Output is otherwise identical (`.md`, `ocr_docling.json`, `text_docling.txt`),
minus the `_ocr.pdf` file.

## Project Structure

```
pdf_ocr/
  run.py           entry point
  config.yaml      pipeline settings
  requirements.txt Python dependencies
  lib/
    ocr_docling.py Docling + Surya pipeline, searchable PDF writer
```

## Dependencies

- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF reading and invisible text layer construction
- [Docling](https://github.com/DS4SD/docling) — layout detection, table extraction, Markdown export
- [Surya OCR](https://github.com/datalab-to/surya) — OCR backend (text detection + recognition, including handwriting)
- [PyYAML](https://pyyaml.org/) — config parsing
- [Transformers](https://huggingface.co/docs/transformers/) — required by Surya models
