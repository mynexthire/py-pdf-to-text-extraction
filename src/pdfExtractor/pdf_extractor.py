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

    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        return "", True

    full_text: list[str] = []
    cv_pages_flagged = 0

    for page_num, page in enumerate(doc):
        all_words = page.get_text("words")

        page_text = _extract_digital_page(page, all_words)
        full_text.append(page_text)

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

            page_area = page.rect.width * page.rect.height
            if page_area > 0 and not has_real_fonts:
                image_area = 0.0
                for info in page.get_image_info():
                    r = fitz.Rect(info["bbox"])
                    image_area += r.width * r.height
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

    doc.close()

    extracted = "\n".join(full_text).strip()

    # All checked pages must be flagged — a single image page (portfolio cover,
    # certificate attachment) among digital pages should not mark the whole CV
    if cv_pages_flagged >= min(_CV_PAGE_LIMIT, page_count):
        # Even if flagged as image-based, enough extracted text means GenAI can still use it
        if len(extracted.split()) >= _MIN_USABLE_WORD_COUNT:
            return extracted, False
        return extracted, True

    return extracted, False


def _extract_digital_page(page, all_words: list[tuple] | None = None) -> str:
    if all_words is None:
        all_words = page.get_text("words")

    page_text = []

    uri_rects = []
    for link in page.get_links():
        if link.get("kind") == fitz.LINK_URI:
            uri = link.get("uri", "").strip()
            if uri:
                uri_rects.append((fitz.Rect(link["from"]), uri))

    blocks = page.get_text(
        "blocks",
        flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
    )
    blocks = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))

    # Pre-build word rects once; reused for every block intersection check
    word_rects = [(w, fitz.Rect(w[:4])) for w in all_words]

    for block in blocks:
        if block[6] != 0:
            continue

        block_text = block[4].strip()
        if not block_text:
            continue

        block_rect = fitz.Rect(block[:4])
        words_in_block = sorted(
            [(w, wr) for w, wr in word_rects if wr.intersects(block_rect)],
            key=lambda x: (x[0][5], x[0][6], x[0][7])
        )

        line_map: dict = {}
        for w, wr in words_in_block:
            x0, y0, x1, y1, word, block_no, line_no, word_no = w
            line_map.setdefault((block_no, line_no), []).append((word_no, word, wr))

        for key in sorted(line_map.keys()):
            line_parts = []
            for word_no, word, word_rect in sorted(line_map[key], key=lambda x: x[0]):
                matched_uris = [
                    uri for lrect, uri in uri_rects if word_rect.intersects(lrect)
                ]
                if matched_uris:
                    line_parts.append(word + "[" + ", ".join(dict.fromkeys(matched_uris)) + "]")
                else:
                    line_parts.append(word)

            page_text.append(" ".join(line_parts))

    return "\n".join(page_text)
