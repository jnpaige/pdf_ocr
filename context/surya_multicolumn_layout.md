# OCR Pipeline — Multi-Column Layout Handling and Engine Options

## Context

This document summarizes development work (May 2026) on the `pdf_ocr` pipeline for OCR of Di Peso's *Casas Grandes* volumes — two-column academic text with figures, tables, and architectural drawings.

---

## Problem 1: Cross-Column Text Merging

The original pipeline used Surya's `RecognitionPredictor` + `DetectionPredictor` directly on full page images. On two-column layouts with adjacent figures, the detection model merged text lines across columns.

Example of merged output:
```
ooting width: .55 .55 .55 1.50 18 .60 GRAND TOTAL 22
Wall thickness: 4 .52 .90 .90 .90 .62
```

## Solution: `layout_rec` Pipeline

Added a `layout_rec` pipeline mode that runs `LayoutPredictor` before recognition. This segments each page into labeled regions (Text, Figure, Table, Caption, etc.), then OCRs each text region independently as a cropped sub-image.

---

## Problem 2: Figure Text Bleeding into Column Crops

With `layout_rec`, a text region's bbox sometimes extends slightly into an adjacent figure area, causing OCR to pick up figure annotations mixed into the column text.

### Approaches tried and rejected

**Post-OCR line filtering (overlap threshold)**: Drop lines with >40% bbox overlap with figure zones. Too aggressive — legitimate boundary lines dropped, causing missing text. Also, adding Figure/Picture to OCR types caused architectural drawings to produce hundreds of tiny fragment boxes.

### Solution: Pre-OCR crop clipping

`_clip_to_exclude_figures()` retracts the text crop's boundary edge to the figure's boundary before OCR. The detector never sees figure content. Figure/Picture bboxes are collected as exclusion zones on every page regardless of whether those types are in `layout_region_types`.

---

## Problem 3: `LayoutPredictor` Initialization

`LayoutPredictor.__init__()` requires `foundation_predictor` as a positional argument. Correct initialization order:

```python
det_predictor        = DetectionPredictor()
foundation_predictor = FoundationPredictor()
layout_predictor     = LayoutPredictor(foundation_predictor)
rec_predictor        = RecognitionPredictor(foundation_predictor)
```

---

## MinerU / PaddleOCR as Alternative Engine

Added `mineru` as a second engine (`ocr_engines: ['mineru']`). Integration: `lib/ocr_mineru.py` calls `magic-pdf` CLI via subprocess, parses `*_middle.json` output, and translates bboxes from PDF points to pixels (`pixel = point * dpi / 72`).

---

## MinerU Setup on Windows — Full Issue Log

Setting up magic-pdf 1.3.12 on Windows 11 with CUDA 12.8 required working through multiple issues in sequence. Each is documented here so future setup is faster.

### Issue 1: `magic-pdf.json` not found

**Error**: `FileNotFoundError: C:\Users\jpaige\magic-pdf.json not found`

**Cause**: MinerU requires a config file in the home directory before it can run. Without it the process exits with code 0 and writes nothing.

**Fix**: Create `C:\Users\jpaige\magic-pdf.json`. See current working config below.

---

### Issue 2: `detectron2` missing

**Error**: `ModuleNotFoundError: No module named 'detectron2'`

**Cause**: When `layout-config` is absent from `magic-pdf.json`, magic-pdf defaults to the LayoutLMv3 layout model, which requires `detectron2`. `detectron2` has no official Windows wheels and is effectively uninstallable on Windows without a complex build.

**Fix**: Add `"layout-config": {"model": "doclayout_yolo"}` to `magic-pdf.json`. DocLayout-YOLO is magic-pdf's newer layout model, ships as a pip package (`doclayout_yolo`), and requires no special dependencies.

---

### Issue 3: OCR model file not found (`ch_PP-OCRv3_det_infer.pth`)

**Error**: `FileNotFoundError: ...\models\OCR\paddleocr_torch\ch_PP-OCRv3_det_infer.pth`

**Cause**: `models_config.yml` inside the installed magic-pdf package hardcodes v3/v4 model filenames, but `opendatalab/PDF-Extract-Kit-1.0` only ships v5 models. Specific mismatches:
- `ch_PP-OCRv3_det_infer.pth` → only `ch_PP-OCRv5_det_infer.pth` present
- `en_PP-OCRv3_det_infer.pth` → not in kit at all
- Many v3 rec models → only v5 equivalents present

**Fix**: Patch `.venv\Lib\site-packages\magic_pdf\model\sub_modules\ocr\paddleocr2pytorch\pytorchocr\utils\resources\models_config.yml`. The patched version is committed to `lib/models_config_patch/models_config.yml`. Key substitutions:
- All `ch_PP-OCRv3_det_infer.pth` → `ch_PP-OCRv5_det_infer.pth`
- `en_PP-OCRv3_det_infer.pth` and `latin` det → `Multilingual_PP-OCRv3_det_infer.pth`
- All v3/v4 rec models → v5 equivalents for languages that have them
- `japan` and `chinese_cht` rec → `ch_PP-OCRv5_rec_infer.pth` (no Japanese/trad-Chinese in kit)

This patch must be reapplied if magic-pdf is reinstalled or upgraded.

---

### Issue 4: `torchvision::nms` CUDA backend not available

**Error**: `NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`

**Cause**: `torch 2.9.0+cu128` is a very recent build. The installed `torchvision 0.24.0` does not have a compiled CUDA NMS kernel for this CUDA version — only the CPU kernel is registered. `pip install torchvision --index-url https://download.pytorch.org/whl/cu128` reports "already satisfied" and does not fix it.

**Fix**: Set `"device-mode": "cpu"` in `magic-pdf.json`. MinerU's YOLO and OCR models run on CPU. Slower but fully functional. A proper fix would require a torchvision cu128 wheel compiled against torch 2.9.0, which may not exist yet.

---

### Issue 5: `UnimerMBartForCausalLM.forward()` unexpected keyword `cache_position`

**Error**: `TypeError: UnimerMBartForCausalLM.forward() got an unexpected keyword argument 'cache_position'`

**Cause**: The formula recognition model (UniMerNet, used for LaTeX equation detection) is incompatible with the installed version of `transformers`, which passes `cache_position` in newer generation calls.

**Fix**: Disable formula recognition entirely in `magic-pdf.json`:
```json
"formula-config": {
    "enable": false
}
```
Formula recognition is only needed for documents with mathematical equations. For text-heavy academic documents like Di Peso, it has no effect on output quality.

---

### Working `magic-pdf.json` (as of May 2026)

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

---

## Surya 0.17.1 Module Structure

Confirmed available submodules:
```
common, detection, foundation, layout, logging, models,
ocr_error, recognition, scripts, settings, table_rec
```

Key imports:
```python
from surya.recognition import RecognitionPredictor
from surya.detection  import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout     import LayoutPredictor
```

`surya.ocr.OCRPredictor` does not exist in Surya 0.17.x.

---

## Configuration Reference

### Surya engine

```yaml
ocr_engines: ['surya']

surya:
  pipeline: 'layout_rec'

  # All types below are OCR'd by default. Remove any to skip.
  # layout_region_types:
  #   - Text
  #   - SectionHeader
  #   - Title
  #   - Caption
  #   - Footnote
  #   - Formula
  #   - ListItem
  #   - PageHeader
  #   - PageFooter
  #   - Table
  #   - Figure
  #   - Picture
```

### MinerU engine

```yaml
ocr_engines: ['mineru']

mineru:
  method: 'ocr'   # 'ocr' for scanned; 'auto'; 'txt' for born-digital
```

---

## Performance Notes

- `layout_rec` is slower than `recognition` — each region is a separate OCR batch call. For 474 pages expect a meaningful increase.
- `layout_region_types` filter speeds things up (e.g. drop `PageHeader`/`PageFooter` to skip running page numbers).
- MinerU `mineru_work/` is cached — re-runs skip the CLI if `*_middle.json` exists. Delete `mineru_work/` to force a re-run.
- MinerU on CPU (required for CUDA 12.8) is slower than GPU but works. A 5-page test PDF takes ~1–2 minutes on CPU.
- Surya detection is calibrated for full-page images; very narrow region crops may produce sub-word detections.