# pdf_ocr

This pipeline converts a corpus of scanned or born-digital PDFs into structured, machine-readable outputs for downstream LLM use — including RAG indexing, agent-crawlable wikis, and full-text search. Each PDF is processed in two complementary passes: Surya OCR reads the raw page imagery (handling both printed text and handwriting), and Docling uses that recognized text alongside its own layout detection to recover document structure — section headers, tables, reading order, multi-column flow — and export clean Markdown. The result per document is a Markdown file suitable for chunking and embedding, a searchable PDF with an invisible text layer, and supporting JSON and plain-text outputs.

## Installation

```powershell
pip install -r requirements.txt
```

For GPU acceleration (recommended for large corpora):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

1. Set `pdf_input` and `output_dir` in `config.yaml`
2. Run:

```powershell
python run.py

# or with a custom config:
python run.py --config path/to/config.yaml
```

## Configuration

```yaml
# config.yaml

pdf_input:  'C:\path\to\pdfs'     # single PDF or directory of PDFs
output_dir: 'C:\path\to\output'   # subfolders created per PDF

docling:
  # markdown_dir: 'C:\path\to\md' # optional: all .md files mirrored here (flat)
```

### `markdown_dir`

When set, every `.md` file is also written to this directory with a flat layout — no subfolders. Useful for pointing a vector store or RAG indexer at a single location:

```yaml
docling:
  markdown_dir: 'C:\corpus\markdown'
```

## Output

For each input PDF a subfolder is created:

```
<output_dir>/<pdf_stem>/
  <pdf_stem>.md        Markdown — section headers, tables, and reading order preserved;
                       primary output for RAG indexing and LLM agent ingestion
  <pdf_stem>_ocr.pdf   Original PDF with an invisible Surya text layer added;
                       fully text-searchable and selectable
  ocr_docling.json     Structured per-page results (text blocks with confidence scores)
  text_docling.txt     Plain text dump, one "=== Page N ===" section per page
```

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
