import logging
import re
from pathlib import Path
from typing import Union
import fitz

_log = logging.getLogger(__name__)

_CIDFONT_RE = re.compile(r'^CIDFont\+F\d+$')

_CV_PAGE_LIMIT = 2
_WORD_COUNT_THRESHOLD = 30
_IMAGE_COVERAGE_THRESHOLD = 0.8
_AVG_WORD_LEN_THRESHOLD = 25
_GARBAGE_CHAR_RATIO_THRESHOLD = 0.4
_OCR_SCORE_THRESHOLD = 50
_MIN_USABLE_WORD_COUNT = 100

_SCORE_LOW_WORD_COUNT = 40
_SCORE_NO_FONTS = 30
_SCORE_HIGH_IMAGE_COVERAGE = 30
_SCORE_LONG_AVG_WORD = 20
_SCORE_HIGH_GARBAGE = 40

# Shared flags for both word and block extraction so a single textpage is reused
_EXTRACT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES


def extract_text(pdf_input: Union[bytes, Path, str]) -> str:
    text, _ = extract(pdf_input)
    return text


def extract(pdf_input: Union[bytes, Path, str]) -> tuple[str, bool]:
    if isinstance(pdf_input, bytes):
        try:
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        except Exception as exc:
            _log.debug("failed to open PDF from bytes: %s", exc)
            return "", True
    else:
        pdf_path = Path(pdf_input)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            _log.debug("failed to open PDF %s: %s", pdf_path, exc)
            return "", True

    try:
        page_count = doc.page_count
        if page_count == 0:
            return "", True

        full_text: list[str] = []
        cv_pages_flagged = 0
        has_any_image = False

        for page_num, page in enumerate(doc):
            # One textpage build shared by both word and block extraction
            tp = page.get_textpage(flags=_EXTRACT_FLAGS)
            all_words = page.get_text("words", textpage=tp)
            all_blocks = page.get_text("blocks", textpage=tp)

            page_text = _extract_digital_page(page, all_words, all_blocks)
            full_text.append(page_text)
            if not has_any_image and page.get_image_info():
                has_any_image = True

            if page_num < _CV_PAGE_LIMIT:
                word_count = len(all_words)
                ocr_score = 0
                page_fonts = doc.get_page_fonts(page_num)

                if word_count < _WORD_COUNT_THRESHOLD:
                    ocr_score += _SCORE_LOW_WORD_COUNT

                if page_fonts:
                    # CIDFont+F<n> are generic wrapper fonts injected by scan/OCR tools, not real embedded text fonts
                    has_real_fonts = any(
                        not _CIDFONT_RE.match(f[3])
                        for f in page_fonts if f[3]
                    )
                else:
                    has_real_fonts = False
                    ocr_score += _SCORE_NO_FONTS

                pr = page.rect
                page_area = (pr.x1 - pr.x0) * (pr.y1 - pr.y0)
                if page_area > 0 and not has_real_fonts:
                    image_area = 0.0
                    for info in page.get_image_info():
                        x0, y0, x1, y1 = info["bbox"]
                        image_area += (x1 - x0) * (y1 - y0)
                    if image_area / page_area > _IMAGE_COVERAGE_THRESHOLD:
                        ocr_score += _SCORE_HIGH_IMAGE_COVERAGE

                if word_count > 0:
                    if sum(len(w[4]) for w in all_words) / word_count > _AVG_WORD_LEN_THRESHOLD:
                        ocr_score += _SCORE_LONG_AVG_WORD

                if page_text:
                    garbage = sum(
                        1 for c in page_text
                        if (ord(c) < 32 and c not in "\n\t\r") or (127 <= ord(c) <= 159)
                    )
                    if garbage / len(page_text) > _GARBAGE_CHAR_RATIO_THRESHOLD:
                        ocr_score += _SCORE_HIGH_GARBAGE

                if ocr_score >= _OCR_SCORE_THRESHOLD:
                    cv_pages_flagged += 1

        extracted = "\n".join(full_text).strip()

        # All checked pages must be flagged — a single image page (portfolio cover,
        # certificate attachment) among digital pages should not mark the whole CV
        if cv_pages_flagged >= min(_CV_PAGE_LIMIT, page_count):
            # Even if flagged as image-based, enough extracted text means GenAI can still use it
            if len(extracted.split()) >= _MIN_USABLE_WORD_COUNT:
                return extracted, False
            return extracted, True

        # Final verdict: image-bearing PDF with too little extractable text → needs OCR
        if has_any_image and len(extracted.split()) < _WORD_COUNT_THRESHOLD:
            return extracted, True

        return extracted, False

    finally:
        doc.close()


def _extract_digital_page(
    page,
    all_words: list[tuple] | None = None,
    all_blocks: list | None = None,
) -> str:
    if all_words is None:
        tp = page.get_textpage(flags=_EXTRACT_FLAGS)
        all_words = page.get_text("words", textpage=tp)
        all_blocks = page.get_text("blocks", textpage=tp)
    elif all_blocks is None:
        all_blocks = page.get_text("blocks", flags=_EXTRACT_FLAGS)

    page_text = []

    # Store link rects as raw float tuples — avoids fitz.Rect overhead during URI matching
    uri_coords: list[tuple] = []
    for link in page.get_links():
        if link.get("kind") == fitz.LINK_URI:
            uri = link.get("uri", "").strip()
            if uri:
                r = link["from"]
                uri_coords.append((r.x0, r.y0, r.x1, r.y1, uri))

    blocks = sorted(all_blocks, key=lambda b: (round(b[1] / 10), b[0]))

    for block in blocks:
        if block[6] != 0:
            continue
        block_text = block[4].strip()
        if not block_text:
            continue

        # Raw float intersection — eliminates fitz.Rect construction + .intersects() overhead
        bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]

        line_map: dict = {}
        for wx0, wy0, wx1, wy1, word, block_no, line_no, word_no in all_words:
            if wx1 > bx0 and bx1 > wx0 and wy1 > by0 and by1 > wy0:
                line_map.setdefault((block_no, line_no), []).append(
                    (word_no, word, wx0, wy0, wx1, wy1)
                )

        for key in sorted(line_map.keys()):
            line_parts = []
            for word_no, word, wx0, wy0, wx1, wy1 in sorted(line_map[key], key=lambda x: x[0]):
                if uri_coords:
                    matched_uris = [
                        uri for ux0, uy0, ux1, uy1, uri in uri_coords
                        if wx1 > ux0 and ux1 > wx0 and wy1 > uy0 and uy1 > wy0
                    ]
                    if matched_uris:
                        line_parts.append(word + "[" + ", ".join(dict.fromkeys(matched_uris)) + "]")
                        continue
                line_parts.append(word)

            page_text.append(" ".join(line_parts))

    return "\n".join(page_text)
