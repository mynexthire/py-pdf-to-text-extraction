import pytest
import fitz
from pathlib import Path
from pdfExtractor import extract_text, extract


@pytest.fixture
def digital_pdf(tmp_path) -> Path:
    """Create a simple digital PDF with known text content."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    page.insert_text((72, 100), "Second line of text")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_link(tmp_path) -> Path:
    """Create a PDF page containing a hyperlink."""
    pdf_path = tmp_path / "linked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Visit example")
    link_rect = fitz.Rect(72, 60, 200, 80)
    page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": "https://example.com"})
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


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


def test_extract_text_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    result = extract_text(pdf_path)
    assert result == ""


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
    assert "Second line" in result


def test_extract_text_bytes_invalid_raises():
    with pytest.raises(Exception):
        extract_text(b"not a pdf")


# ---------------------------------------------------------------------------
# Fixtures for extract() tests
# ---------------------------------------------------------------------------

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
    """PDF with 35 words per page — well above the 30-word scoring threshold."""
    pdf_path = tmp_path / "rich.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # insert_textbox wraps within rect so all words render on the page
    page.insert_textbox(fitz.Rect(50, 50, 545, 500), " ".join(f"word{i}" for i in range(35)))
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# extract() — basic API contract
# ---------------------------------------------------------------------------

def test_extract_returns_tuple(digital_pdf):
    result = extract(digital_pdf)
    assert isinstance(result, tuple) and len(result) == 2


def test_extract_text_part_is_string(digital_pdf):
    text, _ = extract(digital_pdf)
    assert isinstance(text, str)


def test_extract_flag_is_bool(digital_pdf):
    _, flag = extract(digital_pdf)
    assert isinstance(flag, bool)


def test_extract_text_matches_extract_text(digital_pdf):
    assert extract(digital_pdf)[0] == extract_text(digital_pdf)


# ---------------------------------------------------------------------------
# extract() — isImagePresent = False (digital PDFs)
# ---------------------------------------------------------------------------

def test_extract_digital_pdf_not_image(digital_pdf):
    # 5 words (< 30) → +40, but has fonts and no images → total 40 < 50 → False
    _, is_image = extract(digital_pdf)
    assert is_image is False


def test_extract_rich_text_not_image(rich_text_pdf):
    # 35 words → word criterion not triggered → score 0 → False
    _, is_image = extract(rich_text_pdf)
    assert is_image is False


def test_extract_rich_text_content_preserved(rich_text_pdf):
    text, _ = extract(rich_text_pdf)
    assert "word0" in text and "word34" in text


# ---------------------------------------------------------------------------
# extract() — isImagePresent = True (image / scanned PDFs)
# ---------------------------------------------------------------------------

def test_extract_image_pdf_is_image(image_pdf):
    # no words (+40) + no fonts (+30) + image coverage >80% (+30) = 100 ≥ 50 → True
    _, is_image = extract(image_pdf)
    assert is_image is True


def test_extract_image_pdf_text_is_empty(image_pdf):
    text, _ = extract(image_pdf)
    assert text == ""


# ---------------------------------------------------------------------------
# extract() — edge cases
# ---------------------------------------------------------------------------

def test_extract_blank_page_pdf_is_image(tmp_path):
    # No text (+40) and no font resources (+30) = score 70 → flagged as image
    pdf_path = tmp_path / "blank_page.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    text, is_image = extract(pdf_path)
    assert text == "" and is_image is True


def test_extract_accepts_bytes(digital_pdf):
    text, is_image = extract(digital_pdf.read_bytes())
    assert "Hello World" in text
    assert is_image is False


def test_extract_accepts_string_path(digital_pdf):
    text, _ = extract(str(digital_pdf))
    assert "Hello World" in text


def test_extract_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract(Path("/nonexistent/file.pdf"))


def test_extract_invalid_bytes_raises():
    with pytest.raises(Exception):
        extract(b"not a pdf")


# ---------------------------------------------------------------------------
# extract() — multipage majority logic
# ---------------------------------------------------------------------------

def test_extract_image_page_after_first_three_ignored(tmp_path):
    """3 digital CV pages + image page at page 4 → scanned attachment ignored → isImagePresent False."""
    pdf_path = tmp_path / "digital_cv_scanned_attachment.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 500), " ".join(f"word{i}" for i in range(35)))
    # page 4: scanned certificate — should be ignored
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False


def test_extract_image_page_at_three_not_flagged(tmp_path):
    """2 digital pages + image page at page 3 → certificate attachment, not CV → isImagePresent False."""
    pdf_path = tmp_path / "cv_with_scanned_page3.pdf"
    doc = fitz.open()
    for _ in range(2):
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


def test_extract_multipage_image_cv_flagged(tmp_path):
    """1 digital page + 2 image pages → page 1 is digital → isImagePresent False."""
    pdf_path = tmp_path / "mostly_image.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 500), " ".join(f"word{i}" for i in range(35)))
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
        pix.set_rect(pix.irect, (200, 200, 200))
        page.insert_image(fitz.Rect(0, 0, 550, 800), pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    _, is_image = extract(pdf_path)
    assert is_image is False
