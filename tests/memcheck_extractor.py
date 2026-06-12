"""
Memory-leak detector for pdfExtractor.

Two passes:
  1. RSS trend     — run extract() N times on the same file; if RSS keeps
                     climbing after warm-up, something is leaking at the
                     C level (fitz/Pixmap/Tesseract temp buffers).
  2. tracemalloc   — Python-level allocation diff between iteration 5 and
                     iteration N, grouped by source line; shows objects that
                     survive across calls.

Usage:
    python memcheck_extractor.py <pdf> [iterations]
"""

import gc
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, "src")
from pdfExtractor import extract


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def main():
    pdf = Path(sys.argv[1])
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"File: {pdf.name}  ({pdf.stat().st_size / 1024:.0f} KB)")
    print(f"Iterations: {iters}\n")

    # --- Pass 1: RSS trend -------------------------------------------------
    print("iter   RSS(MB)   delta(MB)")
    prev = rss_mb()
    samples = []
    for i in range(1, iters + 1):
        text, status = extract(pdf)
        gc.collect()
        cur = rss_mb()
        samples.append(cur)
        if i <= 5 or i % 5 == 0:
            print(f"{i:4d}   {cur:7.1f}   {cur - prev:+8.2f}")
        prev = cur

    # Linear trend over the post-warm-up half
    tail = samples[len(samples) // 2:]
    growth = (tail[-1] - tail[0]) / max(1, len(tail) - 1)
    print(f"\nextracted {len(text.split())} words, image_based={status}")
    print(f"RSS growth after warm-up: {growth * 1024:+.0f} KB/iteration "
          f"({'LEAK SUSPECTED' if growth > 0.1 else 'stable'})")

    # --- Pass 2: tracemalloc diff -------------------------------------------
    tracemalloc.start(10)
    for _ in range(3):
        extract(pdf)
    gc.collect()
    snap_a = tracemalloc.take_snapshot()

    for _ in range(10):
        extract(pdf)
    gc.collect()
    snap_b = tracemalloc.take_snapshot()
    tracemalloc.stop()

    print("\nTop Python-level allocation growth over 10 extra calls:")
    for stat in snap_b.compare_to(snap_a, "lineno")[:8]:
        if stat.size_diff > 0:
            frame = stat.traceback[0]
            print(f"  {stat.size_diff / 1024:+9.1f} KB  "
                  f"(count {stat.count_diff:+d})  {frame.filename}:{frame.lineno}")


if __name__ == "__main__":
    main()
