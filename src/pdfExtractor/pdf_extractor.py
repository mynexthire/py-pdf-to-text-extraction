import re
from pathlib import Path
from typing import Union
import fitz


def extract_text(pdf_input: Union[bytes, Path, str]) -> str:
    text, _ = extract(pdf_input)
    return text


def extract(pdf_input: Union[bytes, Path, str]) -> tuple[str, bool]:
    if isinstance(pdf_input, bytes):
        try:
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        except Exception:
            return "", True
    else:
        pdf_path = Path(pdf_input)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return "", True

    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        return "", True

    full_text: list[str] = []
    cv_pages_flagged = 0
    cv_page_limit = 2

    for page_num, page in enumerate(doc):
        # Single get_text("words") call — shared by extraction and scoring
        all_words = page.get_text("words")

        page_text = _extract_digital_page(page, all_words)
        full_text.append(page_text)

        if page_num < cv_page_limit:
            word_count = len(all_words)
            ocr_score = 0
            page_fonts = doc.get_page_fonts(page_num)
            # CIDFont+F<n> are generic wrapper fonts injected by scan/OCR tools, not real embedded text fonts
            has_real_fonts = any(
                not re.match(r'^CIDFont\+F\d+$', f[3])
                for f in page_fonts if f[3]
            )

            if word_count < 30:
                ocr_score += 40

            if not page_fonts:
                ocr_score += 30

            page_area = page.rect.width * page.rect.height
            if page_area > 0 and not has_real_fonts:
                image_area = sum(
                    fitz.Rect(info["bbox"]).width * fitz.Rect(info["bbox"]).height
                    for info in page.get_image_info()
                )
                if image_area / page_area > 0.8:
                    ocr_score += 30

            if word_count > 0:
                if sum(len(w[4]) for w in all_words) / word_count > 25:
                    ocr_score += 20

            if page_text:
                garbage = sum(
                    1 for c in page_text
                    if (ord(c) < 32 and c not in "\n\t\r") or (127 <= ord(c) <= 159)
                )
                if garbage / len(page_text) > 0.4:
                    ocr_score += 40

            if ocr_score >= 50:
                cv_pages_flagged += 1

    doc.close()

    extracted = "\n".join(full_text).strip()

    # All checked pages must be flagged — a single image page (portfolio cover,
    # certificate attachment) among digital pages should not mark the whole CV
    if cv_pages_flagged >= min(cv_page_limit, page_count):
        # Even if flagged as image-based, enough extracted text means GenAI can still use it
        if len(extracted.split()) >= 100:
            return extracted, False
        return extracted, True

    return extracted, False


def _extract_digital_page(page, all_words: list | None = None) -> str:
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

    for block in blocks:
        if block[6] != 0:
            continue

        block_text = block[4].strip()
        if not block_text:
            continue

        block_rect = fitz.Rect(block[:4])
        # Filter from pre-fetched words instead of calling get_text per block
        words = sorted(
            [w for w in all_words if fitz.Rect(w[:4]).intersects(block_rect)],
            key=lambda w: (w[5], w[6], w[7])
        )

        line_map = {}
        for x0, y0, x1, y1, word, block_no, line_no, word_no in words:
            line_map.setdefault((block_no, line_no), []).append(
                (word_no, word, fitz.Rect(x0, y0, x1, y1))
            )

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
