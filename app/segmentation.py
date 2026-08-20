import re

# Matches "Part A", "PART - B", "Section I", etc.
PART_RE = re.compile(r"^\s*(PART|SECTION)\s*[-:]?\s*([A-Z]|[IVX]+)\b", re.IGNORECASE)

# Matches top-level question numbers: "1.", "1)", "Q1.", "Q.1"
QNUM_RE = re.compile(r"^\s*(?:Q\.?\s*)?(\d{1,2})[\.\)]\s+")

# Matches subparts: "a)", "(a)", "i)", "(i)"
SUBPART_RE = re.compile(r"^\s*\(?([a-h]|[ivx]{1,4})\)\s+", re.IGNORECASE)

# Matches marks like "(5 marks)", "[5]", "(5M)", "5 Marks"
MARKS_RE = re.compile(r"[\(\[]?\s*(\d{1,3})\s*(?:marks?|m)\s*[\)\]]?", re.IGNORECASE)


def _extract_marks(text: str):
    match = MARKS_RE.search(text)
    return int(match.group(1)) if match else None


def segment_questions(pages: list, source_paper: str = None):
    """
    Convert raw page text into structured question objects.
    Input: list of {page, text, is_scanned} (from PyMuPDF extraction)
    Output: list of question dicts
    """
    questions = []
    current_part = None
    current_q = None  # accumulator for the question being built

    def flush():
        nonlocal current_q
        if current_q and current_q["question_text"].strip():
            current_q["question_text"] = current_q["question_text"].strip()
            current_q["marks"] = _extract_marks(current_q["question_text"])
            current_q["source_paper"] = source_paper
            questions.append(current_q)
        current_q = None

    for page_data in pages:
        if page_data.get("is_scanned"):
            continue  # OCR handles these later

        lines = page_data["text"].split("\n")
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            part_match = PART_RE.match(line)
            if part_match:
                current_part = part_match.group(2).upper()
                continue

            qnum_match = QNUM_RE.match(line)
            subpart_match = SUBPART_RE.match(line)

            if qnum_match:
                flush()
                qnum = qnum_match.group(1)
                remainder = line[qnum_match.end() :]
                current_q = {
                    "question_number": qnum,
                    "subpart": None,
                    "part": current_part,
                    "page": page_data["page"],
                    "question_text": remainder,
                }
            elif subpart_match and current_q:
                # New subpart under the same question number — flush previous subpart, start new
                flush()
                sub = subpart_match.group(1).lower()
                remainder = line[subpart_match.end() :]
                current_q = {
                    "question_number": qnum_match.group(1)
                    if qnum_match
                    else (questions[-1]["question_number"] if questions else None),
                    "subpart": sub,
                    "part": current_part,
                    "page": page_data["page"],
                    "question_text": remainder,
                }
            elif current_q:
                # Continuation of the current question/subpart (multi-line text, code, etc.)
                current_q["question_text"] += " " + line
            # else: stray text before any question number (headers, instructions) — skip

    flush()
    return questions
