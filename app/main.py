import os
import uuid

import pymupdf as fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


def extract_text_by_page(pdf_path: str):
    """Return list of {page, text, is_scanned} for each page in a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        # Heuristic: if almost no extractable text, page is likely scanned
        is_scanned = len(text) < 20
        pages.append({"page": i + 1, "text": text, "is_scanned": is_scanned})
    doc.close()
    return pages


@app.get("/")
def root():
    return {"status": "ok", "message": "Question Paper Analyzer API running"}


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
