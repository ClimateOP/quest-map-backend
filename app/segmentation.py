import re

# Common OCR misreads for digits, scoped to right after "Q" so we don't
# mangle real words elsewhere. e.g. "Ql" -> "Q1", "QO" -> "Q0".
_OCR_DIGIT_FIX = str.maketrans({"l": "1", "I": "1", "i": "1", "O": "0", "o": "0"})

# Matches "Part A", "PART - B", "Section I", etc. Tolerates a few stray
# OCR-noise characters before the keyword (e.g. ": Part A" from a misread rule/border).
PART_RE = re.compile(
    r"^[^A-Za-z0-9]{0,4}\s*(PART|SECTION)\s*[-:]?\s*([A-Z]|[IVX]+)\b", re.IGNORECASE
)

# Matches top-level question numbers: "Q1.", "Q.1", "Q1)", "Q.1)"
# "Q" is REQUIRED — real papers always prefix top-level questions with Q,
# and nested numbered sub-lists inside a subpart (e.g. "1) Write the final
# code... 2) Explain...") never have a Q prefix. Making Q mandatory keeps
# those sub-lists from being mistaken for new top-level questions.
QNUM_RE = re.compile(r"^\s*Q\.?\s*(\d{1,2})[\.\)]\s*", re.IGNORECASE)

# Matches standalone "Q.1" / "Q 1" / "Q1" markers, tolerating OCR digit misreads
# (e.g. "Ql", "QI") right after the "Q".
QNUM_ALT_RE = re.compile(r"^\s*Q\.?\s*([0-9lIiOo]{1,2})\.?\s*", re.IGNORECASE)

# Matches subparts: "a)", "(a)", "i)", "(i)" — anchored to line start OR
# right after a Q-number marker on the same line (e.g. "Q.1 a) Discuss...")
SUBPART_RE = re.compile(r"\(?([a-h]|[ivx]{1,4})\)\s+", re.IGNORECASE)

# Lines that start an instructions block — question-like numbering here
# ("1. Answer five questions...") should NOT be treated as real questions.
INSTRUCTIONS_START_RE = re.compile(r"^\s*Instructions\s*:", re.IGNORECASE)

# Matches marks like "(5 marks)", "[5]", "(5M)", "5 Marks"
MARKS_RE = re.compile(r"[\(\[]?\s*(\d{1,3})\s*(?:marks?|m)\s*[\)\]]?", re.IGNORECASE)

# Matches a bare trailing number at end of line (common in our OCR output,
# where the marks column ends up appended after the question text)
TRAILING_NUM_RE = re.compile(r"(?:^|\s)(\d{1,2})\s*$")


# A question this long almost always means OCR/segmentation boundaries bled
# together (e.g. a missed Q-marker let a later question's text get appended
# to an earlier one). Flag rather than silently trust.
LONG_TEXT_REVIEW_THRESHOLD = 350


def _extract_marks(text: str):
    match = MARKS_RE.search(text)
    if match:
        return int(match.group(1))
    trailing = TRAILING_NUM_RE.search(text)
    if trailing:
        return int(trailing.group(1))
    return None


def segment_questions(pages: list, source_paper: str = None):
    """
    Convert raw page text into structured question objects.
    Input: list of {page, text, is_scanned} (from PyMuPDF extraction)
    Output: list of question dicts
    """
    questions = []
    current_part = None
    current_qnum = None  # tracks the active question number across subparts
    current_q = None  # accumulator for the question being built
    in_instructions = False  # True while inside an "Instructions:" block

    def flush():
        nonlocal current_q
        if current_q and current_q["question_text"].strip():
            current_q["question_text"] = current_q["question_text"].strip()
            current_q["marks"] = _extract_marks(current_q["question_text"])
            current_q["source_paper"] = source_paper
            current_q["needs_review"] = (
                len(current_q["question_text"]) > LONG_TEXT_REVIEW_THRESHOLD
            )
            questions.append(current_q)
        current_q = None

    def start_question(qnum, subpart, page_num, remainder):
        nonlocal current_q, current_qnum
        flush()
        current_qnum = qnum
        current_q = {
            "question_number": qnum,
            "subpart": subpart,
            "part": current_part,
            "page": page_num,
            "question_text": remainder,
        }

    for page_data in pages:
        lines = page_data["text"].split("\n")
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            # Strip stray leading OCR noise (misread table borders/bullets like
            # ")", ":", "|", "~") so every regex below sees a clean line start.
            line = re.sub(r"^[^\w]+", "", line)
            if not line:
                continue

            if INSTRUCTIONS_START_RE.match(line):
                in_instructions = True
                continue

            part_match = PART_RE.match(line)
            if part_match:
                current_part = part_match.group(2).upper()
                in_instructions = False
                continue

            if in_instructions:
                # A real "Q.N" marker always ends the instructions block, even if
                # PART_RE never matched (e.g. OCR garbled the Part heading).
                if QNUM_ALT_RE.match(line) or re.match(
                    r"^\s*Q\.?\s*\d", line, re.IGNORECASE
                ):
                    in_instructions = False
                else:
                    # Bare numbered bullets ("1. Answer five questions...") must
                    # never be mistaken for real questions while still inside this block.
                    continue

            qnum_match = QNUM_RE.match(line) or QNUM_ALT_RE.match(line)
            subpart_match = SUBPART_RE.match(
                line
            )  # anchored below via .match() -> start of line only here

            if qnum_match:
                qnum = qnum_match.group(1).translate(_OCR_DIGIT_FIX)
                rest = line[qnum_match.end() :]
                # Handle "Q.1 a) Discuss..." — subpart marker inline right after the Q-number
                inline_sub = SUBPART_RE.match(rest)
                if inline_sub:
                    start_question(
                        qnum,
                        inline_sub.group(1).lower(),
                        page_data["page"],
                        rest[inline_sub.end() :],
                    )
                else:
                    start_question(qnum, None, page_data["page"], rest)
            elif subpart_match and current_qnum is not None:
                # New subpart under the same question number
                start_question(
                    current_qnum,
                    subpart_match.group(1).lower(),
                    page_data["page"],
                    line[subpart_match.end() :],
                )
            elif current_q:
                # Continuation of the current question/subpart (multi-line text, code, etc.)
                current_q["question_text"] += " " + line
            # else: stray text before any question number (headers, instructions) — skip

    flush()

    # Document-level check: a gap in the Q-number sequence (e.g. Q5 then Q7,
    # no Q6 anywhere) means a question was likely missed entirely during
    # OCR/extraction, not just mis-parsed. Flag every question so a human
    # reviewer knows this paper needs a second look, rather than silently
    # shipping an incomplete question bank.
    seen_numbers = sorted(
        {int(q["question_number"]) for q in questions if q["question_number"]}
    )
    has_gap = any(b - a > 1 for a, b in zip(seen_numbers, seen_numbers[1:]))
    if has_gap:
        for q in questions:
            q["needs_review"] = True
            q["review_reason"] = q.get(
                "review_reason", "possible missing question in sequence"
            )

    return questions
