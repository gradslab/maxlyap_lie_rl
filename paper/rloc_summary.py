"""Small summary of the near-goal shell files on SO(2) (each about 210 MB), read from the directory given as the first
argument; writes results/review/F/rloc_summary_SO2.json.
"""
import glob
import json
import os
import sys

from paths import RES


def largest_radius(shells, ok):
    r = 0.0
    for s in shells:
        if not ok(s):
            break
        r = s["r"]
    return r


out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "SO2__*.json"))):
    d = json.load(open(f))
    sh = d.pop("shells")
    lq, c = d["lambda_min_Q"], d["c"]
    inner = [s for s in sh if s["r"] <= d["r_Nb"]]
    d.update(n_radii=len(sh), r_first=sh[0]["r"], r_last=sh[-1]["r"],
             r_loc_from_shells=largest_radius(sh, lambda s: s["max_ratio"] <= -c),
             r_dec_from_shells=largest_radius(sh, lambda s: s["max_ratio"] < 0),
             r_con_from_shells=largest_radius(sh, lambda s: s["max_contraction"] < 1),
             worst_ratio_in_Nb=max(s["max_ratio"] for s in inner) / lq, ratio_first=sh[0]["max_ratio"] / lq)
    out[os.path.basename(f)[:-5]] = d
    del sh
    print(os.path.basename(f), d["r_loc"], d["n_radii"], flush=True)
json.dump(out, open(RES / "review" / "F" / "rloc_summary_SO2.json", "w"), indent=1)
print(len(out), "files")
