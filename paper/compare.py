"""Compare the regenerated tables and figures in paper/out with the paper's versions in paper/reference."""
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
OUT, REF = HERE / "out", HERE / "reference"


def raster(p, dpi=100):
    pix = fitz.open(p)[0].get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n).astype(int)


bad = 0
for t in sorted(REF.glob("tab_*.tex")):
    o = OUT / t.name
    ok = o.exists() and o.read_text() == t.read_text()
    bad += not ok
    print(f"{t.name:22s} {'identical' if ok else 'DIFFERS'}")
for f in sys.argv[1:] or [p.name for p in (REF / "figs").glob("*.pdf")]:
    o, r = OUT / "figs" / f, REF / "figs" / f
    if not o.exists():
        print(f"{f:22s} not built")
        continue
    a, b = raster(o), raster(r)
    if a.shape != b.shape:
        print(f"{f:22s} DIFFERS in size {a.shape} vs {b.shape}")
        bad += 1
        continue
    d = np.abs(a - b).max(axis=2)
    frac = (d > 24).mean()
    bad += frac > 1e-3
    print(f"{f:22s} {'identical' if frac == 0 else 'DIFFERS' if frac > 1e-3 else 'same'} (pixels differing: {100 * frac:.3f}%)")
sys.exit(1 if bad else 0)
