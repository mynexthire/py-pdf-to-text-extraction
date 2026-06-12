"""
Classification regression check over the CV_PARSING_TEST_CASES corpus.

Expected label is derived from the folder name:
    full_image_cv/, IMAGE_CV/          -> IMAGE_BASED (status True)
    digital_only/, digital_with_images/,
    digital_image_cvs/, test_cv/       -> DIGITAL     (status False)

Usage:
    python corpus_check.py [--ocr] [--only file1.pdf,file2.pdf]

Default runs with ocr=False: page flagging and the all-pages-flagged verdict
are OCR-independent, so this is a fast first pass. Files whose mismatch could
be rescued by OCR text (the sparse-text fallback verdict) should be rerun
with --ocr.

Writes corpus_results.csv (folder, file, expected, got, words, seconds).
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from pdfExtractor import extract

ROOT = Path.home() / "Downloads/CV_PARSING_TEST_CASES"
EXPECT = {
    "full_image_cv": True,
    "IMAGE_CV": True,
    "digital_only": False,
    "digital_with_images": False,
    "digital_image_cvs": False,
    "test_cv": False,
}

use_ocr = "--ocr" in sys.argv
only = None
for i, a in enumerate(sys.argv):
    if a == "--only" and i + 1 < len(sys.argv):
        only = set(sys.argv[i + 1].split(","))

rows = []
mismatches = []
t_start = time.time()

for folder, expected in EXPECT.items():
    pdfs = sorted((ROOT / folder).glob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if p.name in only]
    for pdf in pdfs:
        t0 = time.perf_counter()
        try:
            text, got = extract(pdf, ocr=use_ocr)
            words = len(text.split())
        except Exception as exc:
            text, got, words = "", None, 0
            print(f"ERROR {folder}/{pdf.name}: {exc}", flush=True)
        dt = time.perf_counter() - t0
        rows.append((folder, pdf.name, expected, got, words, round(dt, 2)))
        if got != expected:
            mismatches.append((folder, pdf.name, expected, got, words))
    done = len(rows)
    print(f"[{time.time()-t_start:6.0f}s] finished {folder} "
          f"({len(pdfs)} files, total {done})", flush=True)

out = Path("corpus_results.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "file", "expected_image_based", "got", "words", "seconds"])
    w.writerows(rows)

print(f"\nocr={use_ocr}  results -> {out}")
print(f"{'folder':<22} {'n':>4} {'correct':>8} {'acc':>7}")
total_n = total_ok = 0
for folder, expected in EXPECT.items():
    sub = [r for r in rows if r[0] == folder]
    ok = sum(1 for r in sub if r[3] == r[2])
    total_n += len(sub)
    total_ok += ok
    if sub:
        print(f"{folder:<22} {len(sub):>4} {ok:>8} {ok/len(sub):>6.1%}")
print(f"{'TOTAL':<22} {total_n:>4} {total_ok:>8} {total_ok/max(1,total_n):>6.1%}")

if mismatches:
    print(f"\n{len(mismatches)} mismatches:")
    for folder, name, exp, got, words in mismatches:
        label = lambda v: "IMAGE" if v else "DIGITAL"
        print(f"  {folder}/{name}: expected {label(exp)}, got {label(got)} ({words} words)")
