import io
import os
import re
import uuid

import pymupdf as fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.segmentation import segment_questions

app = FastAPI(title="Question Paper Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OCR_DPI = 300
WORD_RE = re.compile(r"[A-Za-z]{2,}")


def text_quality_score(text: str) -> float:
    """
    Cheap heuristic for embedded-text quality, no LLM needed.
    Returns fraction of the text made up of clean alphabetic words (len >= 2).
    Low score => text layer is likely OCR garbage from a scanning app.
    """
    if not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    clean_words = WORD_RE.findall(text)
    return len(clean_words) / len(words)


MARGIN_QNUM_RE = re.compile(r"^Q?\.?\s*([0-9lIiOoSsZzBbGg]{1,2})$", re.IGNORECASE)
QMARKER_COUNT_RE = re.compile(r"Q\.?\s*\d{1,2}", re.IGNORECASE)

# Common OCR misreads of digits, applied only to short Q-number tokens
# (never to general body text, to avoid corrupting real words).
_OCR_NUM_FIX = str.maketrans(
    {
        "l": "1",
        "I": "1",
        "i": "1",
        "O": "0",
        "o": "0",
        "S": "5",
        "s": "5",
        "Z": "2",
        "z": "2",
        "B": "8",
        "G": "6",
    }
)


def _reconstruct_lines(lines, page_width, expected_q=1):
    """
    Engine-agnostic. Takes a list of {"text", "x0", "y0"} lines (already
    extracted by whichever OCR engine is in use), re-attaches left-margin
    Q-number labels to their nearest body line, and normalizes Q-number
    tokens using sequence continuity (starting from `expected_q`, carried
    across pages by the caller).
    Returns (text, next_expected_q, marker_count).
    """
    margin_labels = []
    body_lines = []
    for line in lines:
        is_margin_position = line["x0"] < page_width * 0.15
        m = MARGIN_QNUM_RE.match(line["text"])
        if is_margin_position and m:
            num = m.group(1).translate(_OCR_NUM_FIX)
            if num.isdigit():
                margin_labels.append({"num": num, "y0": line["y0"]})
                continue
        body_lines.append(line)

    for label in margin_labels:
        if not body_lines:
            continue
        closest = min(body_lines, key=lambda l: abs(l["y0"] - label["y0"]))
        if abs(closest["y0"] - label["y0"]) < 60:  # px tolerance at 300dpi
            closest["text"] = f"Q.{label['num']} {closest['text']}"

    body_lines.sort(key=lambda l: l["y0"])

    inline_q_re = re.compile(r"^(Q\.?\s*)([A-Za-z0-9]{1,2})\b")
    margin_q_re = re.compile(r"^Q\.(\d{1,2})\b")
    out_lines = []
    for l in body_lines:
        text = l["text"]
        mm = margin_q_re.match(text)
        m = mm or inline_q_re.match(text)
        if m:
            token = mm.group(1) if mm else m.group(2)
            translated = token.translate(_OCR_NUM_FIX)
            candidate = int(translated) if translated.isdigit() else None
            # Trust the reading only if it's a plausible next number
            # (equal to, or one past, what we expect); otherwise trust
            # the sequence over the OCR guess.
            if candidate is not None and candidate in (expected_q, expected_q + 1):
                qnum = candidate
            else:
                qnum = expected_q
            text = f"Q.{qnum}" + text[m.end() :]
            expected_q = qnum + 1
        out_lines.append(text)

    marker_count = len(QMARKER_COUNT_RE.findall("\n".join(out_lines)))
    return "\n".join(out_lines), expected_q, marker_count


def _tesseract_lines(img: Image.Image):
    """Run Tesseract and return a list of {"text","x0","y0"} lines."""
    from pytesseract import Output

    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    lines_by_key = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word or data["conf"][i] == "-1":
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        entry = lines_by_key.setdefault(key, {"words": [], "xs": [], "ys": []})
        entry["words"].append(word)
        entry["xs"].append(data["left"][i])
        entry["ys"].append(data["top"][i])

    lines = []
    for entry in lines_by_key.values():
        lines.append(
            {
                "text": " ".join(entry["words"]),
                "x0": min(entry["xs"]),
                "y0": min(entry["ys"]),
            }
        )
    return lines


_paddle_engine = None
_paddle_available = None  # None = not checked yet, True/False once known


def _get_paddle_engine():
    """Lazily create a single shared PaddleOCR instance (loading model
    weights is expensive — do it once per process, not per page)."""
    global _paddle_engine, _paddle_available
    if _paddle_available is False:
        return None
    if _paddle_engine is not None:
        return _paddle_engine
    try:
        from paddleocr import PaddleOCR

        # use_textline_orientation handles rotated/skewed phone photos.
        # PaddleOCR auto-detects and uses a GPU if paddlepaddle-gpu is
        # installed; falls back to CPU otherwise, no code change needed.
        _paddle_engine = PaddleOCR(use_textline_orientation=True, lang="en")
        _paddle_available = True
        print(
            "✅ OCR engine: PaddleOCR (GPU if paddlepaddle-gpu is installed, else CPU)"
        )
        return _paddle_engine
    except Exception as e:
        print(f"⚠️  OCR engine: Tesseract fallback — PaddleOCR unavailable ({e})")
        _paddle_available = False
        return None


@app.on_event("startup")
def _log_ocr_engine_on_startup():
    """Resolve and print which OCR engine is active as soon as the server
    boots, instead of leaving it buried in the first upload's logs."""
    _get_paddle_engine()


def _paddle_lines(img: Image.Image):
    """Run PaddleOCR and return a list of {"text","x0","y0"} lines."""
    import numpy as np

    engine = _get_paddle_engine()
    # PaddleOCR's internal preprocessing (doc unwarping etc.) expects a
    # 3-channel image. Our contrast/sharpen preprocessing step converts to
    # grayscale, which crashes it with a 2D array — force back to RGB.
    result = engine.predict(np.array(img.convert("RGB")))
    lines = []
    for page_result in result:
        texts = page_result.get("rec_texts", [])
        polys = page_result.get("rec_polys", [])
        for text, poly in zip(texts, polys):
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            lines.append({"text": text, "x0": min(xs), "y0": min(ys)})
    return lines


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale + autocontrast + sharpen — cheap, local, no API cost.
    Helps on blurry/low-contrast phone photos."""
    from PIL import ImageFilter, ImageOps

    gray = ImageOps.grayscale(img)
    contrast = ImageOps.autocontrast(gray, cutoff=2)
    return contrast.filter(ImageFilter.SHARPEN)


def ocr_page(page, expected_q: int = 1):
    """
    OCR the page with layout awareness. Many university papers print the
    question number (Q.1, Q.2...) in a separate left-margin column from the
    question body text, which OCR engines tend to read as a separate block,
    detaching "Q.1" from the question it labels. We reconstruct reading
    order by re-attaching each margin question-number label to the body
    line closest to it vertically, then normalize using sequence continuity
    carried in from previous pages.

    Uses PaddleOCR (GPU-accelerated if paddlepaddle-gpu is installed) when
    available — generally more accurate than Tesseract on real-world phone
    photos — and falls back to Tesseract automatically if PaddleOCR isn't
    installed. We try the raw render and a lightly preprocessed version and
    keep whichever recovers more "Q.N" markers.
    Returns (text, next_expected_q).
    """
    pix = page.get_pixmap(dpi=OCR_DPI)
    raw_img = Image.open(io.BytesIO(pix.tobytes("png")))
    page_width = raw_img.width

    engine = _get_paddle_engine()
    get_lines = _paddle_lines if engine is not None else _tesseract_lines

    candidates = []
    for img in (raw_img, _preprocess_for_ocr(raw_img)):
        lines = get_lines(img)
        text, next_q, marker_count = _reconstruct_lines(lines, page_width, expected_q)
        candidates.append((marker_count, text, next_q))

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, best_text, best_next_q = candidates[0]
    return best_text, best_next_q


def extract_text_by_page(pdf_path: str, quality_threshold: float = 0.75):
    """
    Return list of {page, text, is_scanned, source} for each page in a PDF.
    Uses embedded text when it's clean; falls back to our own Tesseract OCR
    when the embedded layer is missing or looks like poor-quality OCR noise
    (common with phone-scanned papers that already carry a bad text layer).
    """
    doc = fitz.open(pdf_path)
    pages = []
    expected_q = 1  # question-number sequence, carried across pages
    for i, page in enumerate(doc):
        embedded_text = page.get_text().strip()
        score = text_quality_score(embedded_text)
        has_image = len(page.get_images()) > 0

        if score < quality_threshold and (has_image or not embedded_text):
            text, expected_q = ocr_page(page, expected_q)
            text = text.strip()
            source = "ocr"
        else:
            text = embedded_text
            source = "embedded"

        pages.append(
            {
                "page": i + 1,
                "text": text,
                "is_scanned": source == "ocr",
                "text_source": source,
            }
        )
    doc.close()
    return pages


@app.get("/")
def root():
    engine = "paddleocr" if _paddle_available else "tesseract"
    return {
        "status": "ok",
        "message": "Question Paper Analyzer API running",
        "ocr_engine": engine,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = {"file_id": file_id, "filename": file.filename, "saved_path": save_path}

    if ext == ".pdf":
        pages = extract_text_by_page(save_path)
        result["pages"] = pages
        result["needs_ocr"] = any(p["is_scanned"] for p in pages)
        result["questions"] = segment_questions(pages, source_paper=file.filename)
    else:
        result["pages"] = None
        result["note"] = "Non-PDF upload — OCR/image handling comes in a later step."

    return result
