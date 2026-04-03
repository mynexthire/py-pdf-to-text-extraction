# pdfExtractor

A lightweight Python library for extracting text from PDF files, built on [PyMuPDF](https://pymupdf.readthedocs.io/).

## Features

- Extract text from digital (text-based) PDFs
- Preserves reading order (top-to-bottom, left-to-right)
- Annotates hyperlinks inline: `word[https://example.com]`

## Installation

```bash
pip install .
```

## Usage

```python
from pathlib import Path
from pdfExtractor import extract_text

text = extract_text(Path("document.pdf"))
print(text)
```

`extract_text` also accepts a plain string path:

```python
text = extract_text("document.pdf")
```

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Requirements

- Python >= 3.10
- pymupdf >= 1.27.2.2

## License

AGPL-3.0 — see [LICENSE](LICENSE) for details.
