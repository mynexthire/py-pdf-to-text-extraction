"""
OCR accuracy bench: compares DPI / PSM / preprocessing variants on image-based CVs.

For each variant, reports:
  - word count
  - mean Tesseract word confidence (image_to_data, conf > 0)
  - presence of known ground-truth strings (name fragments, phone)
  - wall-clock time

Usage:
    python ocr_accuracy_bench.py
"""

import sys
import time
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageOps

CASES = [
    (Path.home() / "Downloads/Resume-3.pdf",
     ["ADIT", "KUMAR", "9658732518"]),
    (Path.home() / "Downloads/pdf r  (2).pdf",
     ["Shreya", "Yadav"]),
]

sys.path.insert(0, "src")
from pdfExtractor.pdf_extractor import _find_image_column_split


def render(page, dpi):
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def prep_none(img):
    return img


def prep_gray_autocontrast(img):
    return ImageOps.autocontrast(img.convert("L"))


def prep_binarize(img):
    g = ImageOps.autocontrast(img.convert("L"))
    return g.point(lambda p: 255 if p > 160 else 0)


PREPS = {
    "raw-rgb": prep_none,
    "gray+autocontrast": prep_gray_autocontrast,
    "binarize@160": prep_binarize,
}


def score_image(img, lang, psm):
    """OCR an image, return (text, mean_conf, n_words)."""
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        img, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )
    words, confs = [], []
    for w, c in zip(data["text"], data["conf"]):
        w = w.strip()
        c = float(c)
        if w and c > 0:
            words.append(w)
            confs.append(c)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return " ".join(words), mean_conf, len(words)


def run_variant(pdf, dpi, psm, prep_name, truths):
    prep = PREPS[prep_name]
    doc = fitz.open(str(pdf))
    t0 = time.perf_counter()
    all_text, all_conf, all_words = [], [], 0
    for page in list(doc)[:2]:
        img = render(page, dpi)
        split = _find_image_column_split(img.convert("L"))
        regions = (
            [img.crop((0, 0, split, img.height)),
             img.crop((split, 0, img.width, img.height))]
            if split else [img]
        )
        for region in regions:
            text, conf, n = score_image(prep(region), "eng", psm)
            all_text.append(text)
            if n:
                all_conf.append((conf, n))
            all_words += n
    doc.close()
    elapsed = time.perf_counter() - t0

    text = " ".join(all_text)
    mean_conf = (
        sum(c * n for c, n in all_conf) / sum(n for _, n in all_conf)
        if all_conf else 0.0
    )
    hits = sum(1 for t in truths if t.lower() in text.lower())
    return mean_conf, all_words, hits, len(truths), elapsed


def main():
    for pdf, truths in CASES:
        if not pdf.exists():
            print(f"SKIP missing {pdf}")
            continue
        print(f"\n=== {pdf.name}  (truth: {truths}) ===")
        print(f"{'dpi':>4} {'psm':>4} {'preprocess':>18} {'conf':>6} "
              f"{'words':>6} {'truth':>6} {'time':>7}")
        for dpi in (200, 300):
            for psm in (6, 4, 3):
                for prep_name in PREPS:
                    conf, n, hits, total, dt = run_variant(
                        pdf, dpi, psm, prep_name, truths)
                    print(f"{dpi:>4} {psm:>4} {prep_name:>18} {conf:>6.1f} "
                          f"{n:>6} {hits}/{total:>4} {dt:>6.1f}s")


if __name__ == "__main__":
    main()
