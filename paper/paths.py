import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "lie_rl" / "results"
OUT = REPO / "paper" / "out"
REF = REPO / "paper" / "reference"
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def rows55():
    rows = []
    for p in sorted(glob.glob(f"{RES}/review/F/search2/*.json")):
        i = os.path.basename(p)[:-5]
        G, d, s, v = i.split("__")
        if v == "base":
            continue
        ev = json.load(open(p))["eval"].split("results/", 1)[1]
        tier = "scratch"
        for r in ("pert_r2", "pert_r3"):
            f = RES / "review" / r / "path" / f"{i}.json"
            if f.exists():
                tier = json.load(open(f))["path"]
        rows.append(dict(id=i, src=str(RES / ev), G=G, alg=d.split("_")[0].upper(), seed=int(s[1:]), v=v, tier=tier))
    assert len(rows) == 55, len(rows)
    return rows
