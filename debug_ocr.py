"""
Diagnostic script: prints the raw lines PaddleOCR detects for a page,
BEFORE our reconstruction/reattachment logic touches them. Run this
locally to see exactly what text/x0/y0 values Paddle is producing.

Usage:
    python debug_ocr.py "path\to\\POC MAY 24.pdf" 1
    (second arg is the 1-indexed page number to inspect, default 1)
"""

import io
import sys

import pymupdf as fitz
from PIL import Image

from app.main import OCR_DPI, _get_paddle_engine, _paddle_lines, _tesseract_lines


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "POC MAY 24.pdf"
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    print(f"Page {page_num}, image size: {img.width}x{img.height}\n")

    engine = _get_paddle_engine()
    if engine is not None:
        print("=== PaddleOCR raw lines (text, x0, y0) ===")
        lines = _paddle_lines(img)
    else:
        print("=== Tesseract raw lines (text, x0, y0) [Paddle unavailable] ===")
        lines = _tesseract_lines(img)

    lines_sorted = sorted(lines, key=lambda l: l["y0"])
    for l in lines_sorted:
        print(f"  y0={l['y0']:5d}  x0={l['x0']:5d}  {l['text']!r}")

    print(f"\nTotal lines detected: {len(lines)}")
    print(f"Page width: {img.width}  (15% mark, old threshold: {img.width * 0.15:.0f})")


if __name__ == "__main__":
    main()
