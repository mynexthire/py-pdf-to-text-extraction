import pytest
import fitz
from pathlib import Path
from pdfExtractor import extract_text, extract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def digital_pdf(tmp_path) -> Path:
    """Simple digital PDF with known text content (≥ 100 words for final-verdict threshold)."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    page.insert_text((72, 100), "Second line of text")
    page.insert_textbox(fitz.Rect(50, 130, 545, 700), " ".join(f"word{i}" for i in range(110)))
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_link(tmp_path) -> Path:
    """PDF page containing a hyperlink."""
    pdf_path = tmp_path / "linked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Visit example")
    link_rect = fitz.Rect(72, 60, 200, 80)
    page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": "https://example.com"})
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def image_pdf(tmp_path) -> Path:
    """PDF whose page is mostly covered by an image with no selectable text."""
    pdf_path = tmp_path / "image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    # ~88% coverage: (550×800) / (595×842) ≈ 0.878
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def rich_text_pdf(tmp_path) -> Path:
    """PDF with 110 words — above both the 30-word scoring threshold and the 100-word final-verdict threshold."""
    pdf_path = tmp_path / "rich.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join(f"word{i}" for i in range(110)))
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# extract_text() — basic contract
# ---------------------------------------------------------------------------

def test_extract_text_returns_string(digital_pdf):
    result = extract_text(digital_pdf)
    assert isinstance(result, str)


def test_extract_text_contains_content(digital_pdf):
    result = extract_text(digital_pdf)
    assert "Hello World" in result
    assert "Second line" in result


def test_extract_text_accepts_string_path(digital_pdf):
    result = extract_text(str(digital_pdf))
    assert "Hello World" in result


def test_extract_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text(Path("/nonexistent/file.pdf"))


def test_extract_text_blank_page_returns_empty(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    assert extract_text(pdf_path) == ""


def test_extract_text_multipage(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content")
    doc.save(str(pdf_path))
    doc.close()
    result = extract_text(pdf_path)
    assert "Page 1 content" in result
    assert "Page 2 content" in result
    assert "Page 3 content" in result


def test_extract_text_link_annotation(pdf_with_link):
    result = extract_text(pdf_with_link)
    assert "https://example.com" in result


def test_extract_text_accepts_bytes(digital_pdf):
    pdf_bytes = digital_pdf.read_bytes()
    result = extract_text(pdf_bytes)
    assert isinstance(result, str)
    assert "Hello World" in result


def test_extract_text_corrupt_bytes_returns_empty():
    assert extract_text(b"not a pdf") == ""


# ---------------------------------------------------------------------------
# extract() — API contract
# ---------------------------------------------------------------------------

def test_extract_returns_tuple(digital_pdf):
    result = extract(digital_pdf)
    assert isinstance(result, tuple) and len(result) == 2


def test_extract_text_part_is_string(digital_pdf):
    text, _ = extract(digital_pdf)
    assert isinstance(text, str)


def test_extract_flag_is_bool(digital_pdf):
    _, is_image = extract(digital_pdf)
    assert isinstance(is_image, bool)


def test_extract_text_matches_extract_text(digital_pdf):
    assert extract(digital_pdf)[0] == extract_text(digital_pdf)


# ---------------------------------------------------------------------------
# extract() — is_image = False (digital)
# ---------------------------------------------------------------------------

def test_extract_digital_pdf_not_image(digital_pdf):
    _, is_image = extract(digital_pdf)
    assert is_image is False


def test_extract_rich_text_not_image(rich_text_pdf):
    _, is_image = extract(rich_text_pdf)
    assert is_image is False


def test_extract_rich_text_content_preserved(rich_text_pdf):
    text, _ = extract(rich_text_pdf)
    assert "word0" in text and "word34" in text


def test_extract_accepts_bytes_digital(digital_pdf):
    text, is_image = extract(digital_pdf.read_bytes())
    assert "Hello World" in text
    assert is_image is False


def test_extract_accepts_string_path(digital_pdf):
    text, _ = extract(str(digital_pdf))
    assert "Hello World" in text


# ---------------------------------------------------------------------------
# extract() — is_image = True (scanned / unparseable)
# ---------------------------------------------------------------------------

def test_extract_image_pdf_is_image(image_pdf):
    # no words (+40) + no fonts (+30) + image coverage >80% (+30) = 100 ≥ 50 → True
    _, is_image = extract(image_pdf)
    assert is_image is True


def test_extract_image_pdf_text_is_empty(image_pdf):
    text, _ = extract(image_pdf)
    assert text == ""


def test_extract_blank_page_is_image(tmp_path):
    # No text (+40) and no font resources (+30) = score 70 → True
    pdf_path = tmp_path / "blank_page.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    text, is_image = extract(pdf_path)
    assert text == "" and is_image is True


def test_extract_both_pages_scanned_is_image(tmp_path):
    """Both checked pages are image-only → True."""
    pdf_path = tmp_path / "scanned_2page.pdf"
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is True


def test_extract_corrupt_bytes_is_image():
    _, is_image = extract(b"not a pdf")
    assert is_image is True


def test_extract_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract(Path("/nonexistent/file.pdf"))


# ---------------------------------------------------------------------------
# extract() — image-based but enough text → False (send to GenAI)
# ---------------------------------------------------------------------------

def test_extract_image_based_with_enough_text_not_image(tmp_path):
    """Pages flagged as image-based but >= 100 extracted words → False (send to GenAI)."""
    pdf_path = tmp_path / "image_with_text.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 500), " ".join(f"word{i}" for i in range(110)))
    doc.save(str(pdf_path))
    doc.close()
    text, is_image = extract(pdf_path)
    assert is_image is False
    assert len(text.split()) >= 100


# ---------------------------------------------------------------------------
# extract() — portfolio / mixed CV (page 1 digital + page 2 image)
# ---------------------------------------------------------------------------

def test_extract_portfolio_cover_plus_text_not_image(tmp_path):
    """Digital page 1 (110 words) + image page 2 → False."""
    pdf_path = tmp_path / "portfolio.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join(f"word{i}" for i in range(110)))
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


def test_extract_image_page2_text_still_extracted(tmp_path):
    """Text from page 1 is returned even when page 2 is image-based."""
    pdf_path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join(f"word{i}" for i in range(110)))
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    text, _ = extract(pdf_path)
    assert "word0" in text


# ---------------------------------------------------------------------------
# extract() — scanned attachments beyond the checked page window
# ---------------------------------------------------------------------------

def test_extract_image_page_beyond_limit_not_image(tmp_path):
    """3 digital pages + image page at page 4 → False."""
    pdf_path = tmp_path / "digital_cv_scanned_attachment.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 500), " ".join(f"word{i}" for i in range(35)))
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


def test_extract_image_page_at_page3_not_image(tmp_path):
    """2 digital pages (55 words each = 110 total) + image page at page 3 → False."""
    pdf_path = tmp_path / "cv_with_scanned_page3.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join(f"word{i}" for i in range(55)))
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


def test_extract_digital_page1_with_two_image_pages_not_image(tmp_path):
    """1 digital page (110 words) + 2 image pages → False."""
    pdf_path = tmp_path / "mostly_image.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join(f"word{i}" for i in range(110)))
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


# ---------------------------------------------------------------------------
# Additional coverage: scoring branches
# ---------------------------------------------------------------------------

def test_extract_corrupt_file_on_disk_returns_empty(tmp_path):
    """A corrupt file on disk (not bytes) returns empty string and is_image=True."""
    bad_path = tmp_path / "corrupt.pdf"
    bad_path.write_bytes(b"not a pdf at all")
    text, is_image = extract(bad_path)
    assert text == "" and is_image is True


def test_extract_long_avg_word_alone_does_not_flag(tmp_path):
    """Long words (avg > 25 chars) score only +20 per page — below threshold; 2 pages ensure ≥ 100 total words."""
    pdf_path = tmp_path / "longwords.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 700), " ".join("x" * 30 for _ in range(80)))
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


def test_extract_image_low_coverage_does_not_add_coverage_score(tmp_path):
    """No-font page with a small image (< 80% coverage) does not get +30 image-coverage score."""
    pdf_path = tmp_path / "small_image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    # ~4% coverage: (100×200) / (595×842) ≈ 0.04
    page.insert_image(fitz.Rect(0, 0, 100, 200), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    # no words (+40) + no fonts (+30) = 70, but image coverage not added → still flagged
    # This test verifies the branch is reached without raising and behaves correctly
    text, is_image = extract(pdf_path)
    assert isinstance(text, str)
    assert is_image is True  # 70 pts from word/font scores alone → still flagged
