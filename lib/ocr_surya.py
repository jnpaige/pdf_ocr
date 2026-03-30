"""Run Surya OCR on a list of page images and return structured results."""
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # large scanned maps exceed PIL's default safety limit


def run_ocr(image_paths: list[Path], languages: list[str]) -> list[dict]:
    """OCR each image with Surya and return one result dict per page.

    Each result dict contains:
      page_index  : int
      image_path  : str
      text_lines  : list of {text, confidence, bbox}
      full_text   : str  (all lines joined with newlines)
    """
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

    print("Loading Surya models...")
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    images = [Image.open(p).convert("RGB") for p in image_paths]

    print(f"Running OCR on {len(images)} page(s)...")
    predictions = rec_predictor(images, det_predictor=det_predictor)

    results = []
    for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
        text_lines = []
        for line in pred.text_lines:
            text_lines.append({
                "text":       line.text,
                "confidence": round(line.confidence, 4),
                "bbox":       line.bbox,
            })
        results.append({
            "page_index": i,
            "image_path": str(img_path),
            "text_lines": text_lines,
            "full_text":  "\n".join(l["text"] for l in text_lines),
        })

    return results
