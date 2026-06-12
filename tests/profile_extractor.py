"""
Profiling script for pdfExtractor.

Runs four analysis passes on a sample of CVs:
  1. Wall-clock timing    — how long extraction takes end-to-end
  2. cProfile hotspots   — which functions consume the most time
  3. Micro-benchmark A   — fitz.Rect intersection vs raw-float intersection
  4. Micro-benchmark B   — shared textpage vs two separate get_text calls

Usage:
    python profile_extractor.py [--files N] [--dir PATH]

Defaults:
    --files 100
    --dir   /home/nikhil/Downloads/CV_PARSING_TEST_CASES
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
import timeit
from pathlib import Path

sys.path.insert(0, "src")
import fitz
from pdfExtractor import extract

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--files", type=int, default=100, metavar="N",
                    help="Number of PDFs to sample (default 100)")
parser.add_argument("--dir", type=Path,
                    default=Path("/home/nikhil/Downloads/CV_PARSING_TEST_CASES"),
                    metavar="PATH")
args = parser.parse_args()

files = sorted(args.dir.rglob("*.pdf"))[: args.files]
if not files:
    sys.exit(f"No PDFs found under {args.dir}")

print(f"Sample: {len(files)} files from {args.dir}\n")


# ---------------------------------------------------------------------------
# 1. Wall-clock timing
# ---------------------------------------------------------------------------

def _run_all():
    for f in files:
        extract(f)

# warm-up — lets fitz load shared libs and avoids cold-cache I/O penalty
for f in files[:5]:
    extract(f)

repeats = 3
elapsed = min(timeit.repeat(_run_all, number=1, repeat=repeats))
ms_per_file = elapsed / len(files) * 1000

print("=" * 60)
print("1. WALL-CLOCK TIMING")
print("=" * 60)
print(f"  {len(files)} files × {repeats} repeats  →  best run: {elapsed:.2f}s")
print(f"  {ms_per_file:.1f} ms / file\n")


# ---------------------------------------------------------------------------
# 2. cProfile — cumulative hotspots
# ---------------------------------------------------------------------------

pr = cProfile.Profile()
pr.enable()
_run_all()
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
ps.print_stats(20)

print("=" * 60)
print("2. cPROFILE  —  top 20 functions by cumulative time")
print("=" * 60)
for line in buf.getvalue().splitlines():
    print(line)
print()


# ---------------------------------------------------------------------------
# Collect realistic sample data for micro-benchmarks
# Words and blocks from actual CVs, loaded once into memory.
# ---------------------------------------------------------------------------

_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

_sample_words: list[tuple] = []
_sample_blocks: list[tuple] = []
_sample_pages_bytes: list[bytes] = []   # raw page bytes for textpage bench

for f in files:
    doc = fitz.open(str(f))
    for page in doc:
        tp = page.get_textpage(flags=_FLAGS)
        w = page.get_text("words", textpage=tp)
        b = page.get_text("blocks", textpage=tp)
        text_blocks = [blk for blk in b if blk[6] == 0 and blk[4].strip()]
        if w and text_blocks:
            _sample_words.extend(w)
            _sample_blocks.extend(text_blocks)
            # Capture this page as a standalone single-page PDF for section 4
            if len(_sample_pages_bytes) < 20:
                mini = fitz.open()
                mini.insert_pdf(doc, from_page=page.number, to_page=page.number)
                _sample_pages_bytes.append(mini.tobytes())
                mini.close()
        if len(_sample_words) >= 5000 and len(_sample_pages_bytes) >= 20:
            break
    doc.close()
    if len(_sample_words) >= 5000 and len(_sample_pages_bytes) >= 20:
        break

_sample_words  = _sample_words[:3000]
_sample_blocks = _sample_blocks[:60]

print(f"  (collected {len(_sample_words)} words, "
      f"{len(_sample_blocks)} blocks, "
      f"{len(_sample_pages_bytes)} pages for micro-benchmarks)\n")


# ---------------------------------------------------------------------------
# 3. Micro-benchmark A: fitz.Rect intersection vs raw-float intersection
#
#    This is the hottest inner loop: for every block on a page, every word
#    on that page is tested for spatial intersection with the block rect.
#    The old code built a fitz.Rect per word (O(words) allocations) and then
#    called .intersects() per (word, block) pair.  The new code unpacks the
#    four floats already present in the word tuple and compares directly.
# ---------------------------------------------------------------------------

def bench_fitz_rect():
    """Old approach: allocate fitz.Rect per word, call .intersects() per pair."""
    word_rects = [(w, fitz.Rect(w[:4])) for w in _sample_words]
    for block in _sample_blocks:
        block_rect = fitz.Rect(block[:4])
        [w for w, wr in word_rects if wr.intersects(block_rect)]


def bench_raw_float():
    """New approach: unpack x0/y0/x1/y1 already in the tuple, compare floats."""
    for block in _sample_blocks:
        bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
        [
            w for w in _sample_words
            if w[2] > bx0 and bx1 > w[0] and w[3] > by0 and by1 > w[1]
        ]


N = 200
t_rect = min(timeit.repeat(bench_fitz_rect, number=N, repeat=5)) / N * 1000
t_raw  = min(timeit.repeat(bench_raw_float,  number=N, repeat=5)) / N * 1000
speedup_a = t_rect / t_raw if t_raw > 0 else float("inf")

print("=" * 60)
print("3. MICRO-BENCHMARK A  —  fitz.Rect vs raw-float intersection")
print(f"   ({len(_sample_words)} words  ×  {len(_sample_blocks)} blocks per iteration)")
print("=" * 60)
print(f"  fitz.Rect approach : {t_rect:6.2f} ms")
print(f"  raw-float approach : {t_raw:6.2f} ms")
print(f"  speedup            : {speedup_a:.1f}×\n")


# ---------------------------------------------------------------------------
# 4. Micro-benchmark B: shared textpage vs two separate get_text calls
#
#    Each page needs both a word list (for scoring) and a block list (for
#    text reconstruction).  The old code called get_text("words") and
#    get_text("blocks") independently — each triggers a full MuPDF textpage
#    build.  The new code calls get_textpage() once and passes the result to
#    both get_text calls, halving the textpage-build work.
#
#    Pages are pre-loaded as bytes so doc.open/close overhead doesn't
#    swamp the signal we're measuring.
# ---------------------------------------------------------------------------

def bench_two_calls():
    """Old: two independent get_text calls → two textpage builds."""
    for pdf_bytes in _sample_pages_bytes:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        page.get_text("words")
        page.get_text("blocks", flags=_FLAGS)
        doc.close()


def bench_shared_tp():
    """New: one get_textpage(), reused for both words and blocks."""
    for pdf_bytes in _sample_pages_bytes:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        tp = page.get_textpage(flags=_FLAGS)
        page.get_text("words",  textpage=tp)
        page.get_text("blocks", textpage=tp)
        doc.close()


N = 30
t_two    = min(timeit.repeat(bench_two_calls, number=N, repeat=5)) / N / len(_sample_pages_bytes) * 1000
t_shared = min(timeit.repeat(bench_shared_tp,  number=N, repeat=5)) / N / len(_sample_pages_bytes) * 1000
speedup_b = t_two / t_shared if t_shared > 0 else float("inf")

print("=" * 60)
print("4. MICRO-BENCHMARK B  —  shared textpage vs two separate calls")
print(f"   ({len(_sample_pages_bytes)} pages, timed per page)")
print("=" * 60)
print(f"  two get_text calls : {t_two:6.3f} ms / page")
print(f"  shared textpage    : {t_shared:6.3f} ms / page")
print(f"  speedup            : {speedup_b:.1f}×\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  End-to-end throughput  : {ms_per_file:.1f} ms / file")
print(f"  Intersection speedup   : {speedup_a:.1f}×  (raw float vs fitz.Rect)")
print(f"  Textpage speedup       : {speedup_b:.1f}×  (shared vs two calls)")
