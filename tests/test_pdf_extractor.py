import pytest
import fitz
from pathlib import Path
from pdfExtractor import extract_text


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
    assert "Hello World" in result
