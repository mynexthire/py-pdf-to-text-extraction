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
_SCORE_ZERO_WORDS = 50       # a wordless page can only yield text via OCR — flag outright
_SCORE_NO_FONTS = 30
_SCORE_HIGH_IMAGE_COVERAGE = 30
_SCORE_LONG_AVG_WORD = 20
_SCORE_HIGH_GARBAGE = 40
_SCORE_HIGH_DRAWINGS = 20
_HIGH_DRAWING_COUNT = 500

_OCR_DPI = 300               # 300 beats 200 by ~3.5 mean-confidence points on the CV corpus
_OCR_MIN_WORDS = 10
_OCR_BUDGET = 3
_OCR_LANG = "eng"
_OCR_PSM = "--psm 4"         # single column, variable sizes — recovers name headers psm 6 splits
_OCR_MIN_CONF = 10           # decorative glyphs OCR at conf ~0; real words rarely drop below ~25
_OCR_BULLET_GAP_RATIO = 1.5  # leading <=2-char word with gap > ratio x height = bullet glyph, not text
_OCR_COLUMN_SAMPLES = 40     # vertical strips to probe when hunting for a column gutter
_OCR_GUTTER_THRESHOLD = 0.15 # gutter must have < 15% of the darkest strip's pixel count

# Shared flags for both word and block extraction so a single textpage is reused
_EXTRACT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

try:
    import pytesseract
    from PIL import Image, ImageOps
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def extract_text(pdf_input: Union[bytes, Path, str]) -> str:
    text, _ = extract(pdf_input)
    return text


def extract(
    pdf_input: Union[bytes, Path, str],
    *,
    ocr: bool = True,
    ocr_lang: str = _OCR_LANG,
    ocr_dpi: int = _OCR_DPI,
) -> tuple[str, bool]:
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
        digital_word_total = 0
        ocr_budget = _OCR_BUDGET

        for page_num, page in enumerate(doc):
            # One textpage build shared by both word and block extraction
            tp = page.get_textpage(flags=_EXTRACT_FLAGS)
            all_words = page.get_text("words", textpage=tp)
            all_blocks = page.get_text("blocks", textpage=tp)
            del tp  # all_words/all_blocks are plain Python lists; release C-level TextPage now

            page_text = _extract_digital_page(page, all_words, all_blocks)
            digital_word_total += len(all_words)
            # get_images() enumerates image xrefs without decoding them —
            # scans with corrupt image streams (zlib errors) come back empty
            # from get_image_info() and would otherwise read as image-free
            if not has_any_image and page.get_images():
                has_any_image = True

            is_page_image_based = False
            if page_num < _CV_PAGE_LIMIT:
                word_count = len(all_words)
                ocr_score = 0
                page_fonts = doc.get_page_fonts(page_num)

                # Cheap, data-already-in-hand signals first. The expensive page
                # scans below (get_image_info / get_drawings) are only worth
                # paying for when they could still tip the score past the
                # threshold — a healthy digital page never reaches them.
                if word_count == 0:
                    ocr_score += _SCORE_ZERO_WORDS
                elif word_count < _WORD_COUNT_THRESHOLD:
                    ocr_score += _SCORE_LOW_WORD_COUNT

                if page_fonts:
                    # CIDFont+F<n> are generic wrapper fonts injected by scan/OCR tools, not real embedded text fonts
                    has_real_fonts = any(
                        not _CIDFONT_RE.match(f[3])
                        for f in page_fonts if f[3]
                    )
                else:
                    has_real_fonts = False
                if not has_real_fonts:
                    ocr_score += _SCORE_NO_FONTS

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

                # Image coverage requires get_image_info() (decodes image geometry).
                # Skip it once the page is already flagged — the extra points are
                # never read, only the >= threshold decision is.
                if ocr_score < _OCR_SCORE_THRESHOLD:
                    pr = page.rect
                    page_area = (pr.x1 - pr.x0) * (pr.y1 - pr.y0)
                    if page_area > 0:
                        image_area = 0.0
                        for info in page.get_image_info():
                            x0, y0, x1, y1 = info["bbox"]
                            image_area += (x1 - x0) * (y1 - y0)
                        if image_area / page_area > _IMAGE_COVERAGE_THRESHOLD:
                            ocr_score += _SCORE_HIGH_IMAGE_COVERAGE

                # Vector-design CVs (Illustrator/Figma exports) store content as paths, not text or images.
                # Hundreds of drawings with few extractable words is the reliable signal.
                # get_drawings() parses the full content stream and is the most
                # expensive call here — only run it when its +20 could actually
                # reach the threshold (i.e. the score is already within 20 of it
                # but not yet over). Normal digital pages sit at 0 and skip it.
                if _OCR_SCORE_THRESHOLD - _SCORE_HIGH_DRAWINGS <= ocr_score < _OCR_SCORE_THRESHOLD:
                    if len(page.get_drawings()) > _HIGH_DRAWING_COUNT:
                        ocr_score += _SCORE_HIGH_DRAWINGS

                if ocr_score >= _OCR_SCORE_THRESHOLD:
                    cv_pages_flagged += 1
                    is_page_image_based = True
            elif cv_pages_flagged >= min(_CV_PAGE_LIMIT, page_count):
                # All checked pages were image-based → treat remaining pages the same
                is_page_image_based = True

            # Layout signals (image coverage, drawings) can flag pages that
            # still carry healthy digital text — OCR would replace good text
            # with an approximation and burn seconds. Only OCR when the
            # digital text is genuinely unusable.
            if (is_page_image_based and ocr and _OCR_AVAILABLE and ocr_budget > 0
                    and not _digital_text_usable(page_text, all_words)):
                ocr_text = _ocr_page(page, lang=ocr_lang, dpi=ocr_dpi)
                ocr_budget -= 1
                if len(ocr_text.split()) >= _OCR_MIN_WORDS:
                    page_text = ocr_text

            full_text.append(page_text)

        extracted = "\n".join(full_text).strip()

        # All checked pages must be flagged — a single image page (portfolio cover,
        # certificate attachment) among digital pages should not mark the whole CV
        if cv_pages_flagged >= min(_CV_PAGE_LIMIT, page_count):
            # Portfolios and vector-design CVs can flag both checked pages on
            # layout signals while later pages carry plenty of real digital text
            if digital_word_total >= _MIN_USABLE_WORD_COUNT:
                return extracted, False
            return extracted, True

        # Final verdict: image-bearing PDF with too little extractable text → needs OCR
        if has_any_image and len(extracted.split()) < _WORD_COUNT_THRESHOLD:
            return extracted, True

        return extracted, False

    finally:
        doc.close()
        # MuPDF keeps a process-wide resource store (decoded images, glyphs,
        # ~256 MB default ceiling) that survives doc.close(). Cross-document
        # reuse is worthless for one-shot CV extraction, and in long-running
        # services the store reads as a ~10 MB/call leak on the OCR path.
        fitz.TOOLS.store_shrink(100)


def _digital_text_usable(page_text: str, all_words: list) -> bool:
    """True when a page's extractable text is healthy enough to keep as-is.

    Mirrors the text-quality scoring signals (word count, average word length,
    garbage-character ratio) — but not the layout signals, which justify
    flagging a page image-based without implying its text is bad.
    """
    if len(all_words) < _WORD_COUNT_THRESHOLD:
        return False
    if sum(len(w[4]) for w in all_words) / len(all_words) > _AVG_WORD_LEN_THRESHOLD:
        return False
    if page_text:
        garbage = sum(
            1 for c in page_text
            if (ord(c) < 32 and c not in "\n\t\r") or (127 <= ord(c) <= 159)
        )
        if garbage / len(page_text) > _GARBAGE_CHAR_RATIO_THRESHOLD:
            return False
    return True


def _find_image_column_split(img) -> int | None:
    """Return x pixel where a column gutter sits, or None for single-column pages.

    Scans vertical strips across the middle 20-80% of the image width and picks
    the strip with the fewest dark pixels — that's the gutter between columns.
    Only returns a split when the gutter is significantly emptier than the content
    columns (_OCR_GUTTER_THRESHOLD controls how empty "empty" must be).
    """
    width, height = img.size
    search_start = int(width * 0.2)
    search_end = int(width * 0.8)
    strip_w = max(10, (search_end - search_start) // _OCR_COLUMN_SAMPLES)

    counts, positions = [], []
    for x in range(search_start, search_end, strip_w):
        end_x = min(x + strip_w, width)
        hist = img.crop((x, 0, end_x, height)).histogram()
        dark = sum(hist[:128])  # pixel values 0-127 = dark content
        counts.append(dark)
        positions.append(x + strip_w // 2)

    if not counts:
        return None

    min_dark = min(counts)
    max_dark = max(counts)
    if max_dark == 0 or min_dark >= max_dark * _OCR_GUTTER_THRESHOLD:
        return None

    return positions[counts.index(min_dark)]


def _assemble_ocr_text(data: dict) -> str:
    """Rebuild reading-order text from image_to_data output.

    Drops words below _OCR_MIN_CONF (decorative graphics OCR as conf-0 garbage)
    and leading bullet glyphs, which Tesseract reads as 'e'/'¢' but betray
    themselves geometrically: a <=2-char first word followed by a gap far wider
    than normal word spacing.
    """
    rows: dict = {}
    order: list = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word or float(data["conf"][i]) < _OCR_MIN_CONF:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in rows:
            rows[key] = []
            order.append(key)
        rows[key].append(i)

    pieces: list = []
    prev_para = None
    for key in order:
        idxs = rows[key]
        if len(idxs) >= 2:
            first, second = idxs[0], idxs[1]
            gap = data["left"][second] - (data["left"][first] + data["width"][first])
            height = data["height"][first] or 1
            if len(data["text"][first].strip()) <= 2 and gap > _OCR_BULLET_GAP_RATIO * height:
                idxs = idxs[1:]
        para = key[:2]
        if prev_para is not None and para != prev_para:
            pieces.append("")
        pieces.append(" ".join(data["text"][i].strip() for i in idxs))
        prev_para = para
    return "\n".join(pieces).strip()


def _ocr_page(page, lang: str = _OCR_LANG, dpi: int = _OCR_DPI) -> str:
    if not _OCR_AVAILABLE:
        return ""
    pix = None
    try:
        # Render RGB — rendering grayscale at the MuPDF level loses
        # white-on-color name headers
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        del pix
        pix = None

        # Luminance conversion must happen post-render: histogram()[:128] on
        # an RGB image counts only the red band, skewing gutter detection
        gray = img.convert("L")
        img.close()

        try:
            split = _find_image_column_split(gray)
        except Exception:
            split = None

        if split:
            regions = [
                gray.crop((0, 0, split, gray.height)),
                gray.crop((split, 0, gray.width, gray.height)),
            ]
        else:
            regions = [gray]

        texts = []
        for region in regions:
            # Per-region autocontrast lifts low-contrast text (light-on-light
            # headers) before Tesseract's internal binarization.
            # image_to_data over image_to_string: same recognition pass, but
            # exposes per-word confidence and boxes for garbage filtering.
            data = pytesseract.image_to_data(
                ImageOps.autocontrast(region), lang=lang, config=_OCR_PSM,
                output_type=pytesseract.Output.DICT,
            )
            texts.append(_assemble_ocr_text(data))
        return "\n".join(t for t in texts if t).strip()

    except Exception as exc:
        _log.warning("OCR failed on page %s: %s", page.number, exc)
        return ""
    finally:
        if pix is not None:
            del pix


def _detect_column_split(text_blocks: list, page_width: float) -> float | None:
    """Return x split point for two-column layout, or None if single column."""
    x0_vals = sorted(set(round(b[0]) for b in text_blocks))
    if len(x0_vals) < 2:
        return None
    max_gap, split = 0, None
    for i in range(1, len(x0_vals)):
        gap = x0_vals[i] - x0_vals[i - 1]
        if gap > max_gap:
            max_gap = gap
            split = (x0_vals[i] + x0_vals[i - 1]) / 2
    return split if max_gap > page_width * 0.1 else None


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

    text_blocks = [b for b in all_blocks if b[6] == 0]
    split = _detect_column_split(text_blocks, page.rect.width)

    if split is not None:
        left = sorted([b for b in text_blocks if b[0] < split], key=lambda b: b[1])
        right = sorted([b for b in text_blocks if b[0] >= split], key=lambda b: b[1])
        blocks = left + right
    else:
        blocks = sorted(text_blocks, key=lambda b: (round(b[1] / 10), b[0]))

    # Bucket words by their native block number once (O(words)) instead of
    # re-scanning every word against every block rect (O(blocks × words)).
    # get_text("words") and get_text("blocks") share the same textpage block
    # numbering, so a word's block_no (index 5) is exactly the block it belongs
    # to. The old geometric intersection rediscovered that — but emitted a word
    # once per block rect it overlapped, so stacked/overlapping blocks (e.g.
    # enhancv CV templates) duplicated header text. Native block_no assigns each
    # word to exactly one block, killing the duplication.
    words_by_block: dict = {}
    for w in all_words:
        words_by_block.setdefault(w[5], []).append(w)

    for block in blocks:
        block_text = block[4].strip()
        if not block_text:
            continue

        line_map: dict = {}
        for wx0, wy0, wx1, wy1, word, block_no, line_no, word_no in words_by_block.get(block[5], ()):
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
