# pdf_ocr

A pipeline for converting scanned PDFs into searchable PDFs with selectable text. Supports two OCR engines: [Surya OCR](https://github.com/datalab-to/surya) and [MinerU / PaddleOCR](https://github.com/opendatalab/MinerU).

## What It Does

1. **Renders** each page of a PDF to a PNG image at a configurable DPI
2. **OCRs** each page using the configured engine, producing structured JSON with per-line text, confidence scores, and bounding boxes
3. **Builds** a searchable PDF with an invisible text layer overlaid on the original page images — visually identical to the scan but fully text-searchable and selectable

## Requirements

- Python 3.10+
- PyTorch (GPU recommended; CPU works but is slow)
- Dependencies in `requirements.txt`

### Install (Surya engine)

```powershell
pip install -r requirements.txt

# For GPU acceleration (recommended), install the CUDA build of PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Install (MinerU engine — optional)

**1. Install the package**

```powershell
pip install "magic-pdf[full]"
```

**2. Install PaddlePaddle with GPU support** (optional)

`magic-pdf[full]` pulls in the CPU build of PaddlePaddle. For GPU acceleration:

```powershell
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Note: PaddlePaddle GPU wheels currently support up to ~CUDA 12.3. If your CUDA version is newer, set `"device-mode": "cpu"` in `magic-pdf.json` (see step 4).

**3. Download model weights**

MinerU requires model files downloaded locally before first use. Without them `magic-pdf` exits silently.

```powershell
pip install huggingface_hub

# Main extraction kit (~5 GB)
hf download opendatalab/PDF-Extract-Kit-1.0 --local-dir "C:\Users\jpaige\magic-pdf-models\PDF-Extract-Kit-1.0"

# Layout reader model (~700 MB)
hf download hantian/layoutreader --local-dir "C:\Users\jpaige\magic-pdf-models\layoutreader"
```

**4. Create the MinerU config file**

Create `C:\Users\jpaige\magic-pdf.json`:

```json
{
    "models-dir": "C:\\Users\\jpaige\\magic-pdf-models\\PDF-Extract-Kit-1.0\\models",
    "layoutreader-model-dir": "C:\\Users\\jpaige\\magic-pdf-models\\layoutreader",
    "device-mode": "cpu",
    "layout-config": {
        "model": "doclayout_yolo"
    },
    "formula-config": {
        "enable": false
    }
}
```

Key notes:
- `models-dir` points to the `models` subdirectory *inside* the PDF-Extract-Kit folder, not the root
- `"device-mode": "cpu"` is required on CUDA 12.8+ because torchvision's CUDA NMS kernel is not available for that version; MinerU falls back to CPU cleanly
- `layout-config.model: doclayout_yolo` uses the newer YOLO-based layout model and avoids the `detectron2` dependency (which has no Windows wheels)
- `formula-config.enable: false` disables LaTeX formula recognition, which crashes due to a `transformers` version incompatibility — not needed for text documents

**5. Patch the OCR model filename config**

magic-pdf 1.3.12 ships a `models_config.yml` that references v3/v4 model filenames, but PDF-Extract-Kit-1.0 only includes v5 models. A patched copy is already committed to this repo at:

```
lib/models_config_patch/models_config.yml
```

Copy it over the installed version:

```powershell
Copy-Item "lib\models_config_patch\models_config.yml" `
  ".venv\Lib\site-packages\magic_pdf\model\sub_modules\ocr\paddleocr2pytorch\pytorchocr\utils\resources\models_config.yml"
```

> This patch replaces `ch_PP-OCRv3_det_infer.pth` → `ch_PP-OCRv5_det_infer.pth` throughout, and updates English/Latin det to `Multilingual_PP-OCRv3_det_infer.pth` and all v3/v4 rec models to their v5 equivalents, matching what PDF-Extract-Kit-1.0 actually ships.

**6. Verify**

```powershell
magic-pdf -p path\to\test.pdf -o C:\temp\mineru_test -m ocr
```

A `*_middle.json` file should appear under `C:\temp\mineru_test\<pdf_stem>\ocr\`. If nothing is written, run with full terminal output to see the error:

```powershell
magic-pdf -p path\to\test.pdf -o C:\temp\mineru_test -m ocr 2>&1
```

## Usage

1. Edit `config.yaml` to set your input PDF path (or directory of PDFs), output directory, and engine
2. Run the pipeline:

```powershell
python run.py
python run.py --config path/to/config.yaml
```

### Page Selection UI

```powershell
python select_pages.py
```

Opens a scrollable 3-column grid of page thumbnails. Left-click to preview a page full-size; right-click or use the checkbox to include/exclude it. Save the selection back to `config.yaml`, then run the pipeline on selected pages only.

### Output Structure

For each input PDF, a subfolder is created in the output directory:

```
<pdf_stem>_<dpi>dpi_<engine>/
  page_000.png                 # rendered page images
  page_001.png
  ...
  ocr_<engine>.json            # structured OCR results (text, bboxes, confidence)
  text_<engine>.txt            # plain text output
  <pdf_stem>_ocr_<engine>.pdf  # searchable PDF
  mineru_work/                 # MinerU intermediate output (mineru engine only)
```

## Configuration

All settings are in `config.yaml`:

```yaml
# Input — single PDF or directory of PDFs
pdf_input: 'C:\path\to\your\pdfs'

# Output directory
output_dir: 'C:\path\to\output'

# Page rendering resolution
page_image_dpi: 150

# OCR engine — 'surya' or 'mineru'
ocr_engines: ['surya']
ocr_languages: ['en']

# Surya pipeline settings (surya engine only)
surya:
  pipeline: 'layout_rec'

# MinerU settings (mineru engine only)
# mineru:
#   method: 'ocr'   # 'ocr' for scanned PDFs; 'auto' to let MinerU decide
```

## OCR Engines

### Surya (`ocr_engines: ['surya']`)

Runs locally using PyTorch. Three pipeline modes via `surya.pipeline`:

**`recognition`** — Runs text detection and recognition directly on the full page image. Fast, no layout awareness. Lines from different columns or adjacent figures may merge on complex pages.

**`layout_rec`** (recommended for multi-column documents) — Runs Surya's layout analysis first to segment each page into labeled regions (Text, Figure, Table, Caption, etc.), then OCRs each region independently as a cropped sub-image. Text crops are clipped to exclude overlapping Figure/Picture areas before OCR, preventing figure annotations from bleeding into column results.

**`ocr`** — Uses Surya's high-level `OCRPredictor` if available. Does not exist in Surya 0.17.x.

### Filtering Layout Regions (layout_rec only)

All region types are OCR'd by default. Remove types you want to skip:

```yaml
surya:
  pipeline: 'layout_rec'
  layout_region_types:
    - Text
    - SectionHeader
    - Title
    - Caption
    - Footnote
    - Formula
    - ListItem
    - PageHeader     # remove to suppress running page numbers
    - PageFooter     # remove to suppress running page numbers
    - Table
    - Figure
    - Picture
```

### MinerU / PaddleOCR (`ocr_engines: ['mineru']`)

Runs locally using PaddlePaddle + PyTorch. MinerU takes the PDF directly, runs its own layout detection (DocLayout-YOLO) and PaddleOCR pipeline. Results are translated from PDF point coordinates to pixel coordinates matching the rendered page images.

Intermediate outputs are written to `mineru_work/` and reused on re-runs (the CLI call is skipped if `*_middle.json` already exists). To force a re-run, delete `mineru_work/`.

```yaml
ocr_engines: ['mineru']
mineru:
  method: 'ocr'   # 'ocr' for scanned PDFs; 'auto' lets MinerU decide;
                  # 'txt' for born-digital PDFs (extracts embedded text, no OCR)
```

## Project Structure

```
pdf_ocr/
  run.py                       # main entry point
  select_pages.py              # interactive page selection UI
  config.yaml                  # all pipeline settings
  requirements.txt             # Python dependencies
  lib/
    render.py                  # renders PDF pages to PNG via PyMuPDF
    ocr_surya.py               # Surya OCR wrapper
    ocr_mineru.py              # MinerU / PaddleOCR wrapper
    pdf_writer.py              # builds searchable PDF
    models_config_patch/
      models_config.yml        # patched OCR model filename map for magic-pdf 1.3.x
  context/                     # development notes and decision context
```

## Dependencies

### Surya engine
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF rendering and searchable PDF construction
- [Surya OCR](https://github.com/datalab-to/surya) (0.17.x) — text detection, recognition, layout analysis
- [Pillow](https://pillow.readthedocs.io/) — image handling
- [PyYAML](https://pyyaml.org/) — config file parsing
- [Transformers](https://huggingface.co/docs/transformers/) — required by Surya models

### MinerU engine (optional)
- [MinerU / magic-pdf](https://github.com/opendatalab/MinerU) — layout detection + PaddleOCR pipeline
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en) — PaddleOCR backend

## Notes

- Tested with Surya 0.17.1 and magic-pdf 1.3.12 on Windows 11 with CUDA 12.8
- The searchable PDF uses invisible text (render mode 3) with horizontal scaling to match bounding boxes — the same approach used by ocrmypdf
- Large scanned images are supported (PIL pixel limit is disabled)
- Pages already rendered to PNG are skipped on re-runs
- MinerU runs are cached: if `mineru_work/*_middle.json` exists the CLI is skipped on re-runs