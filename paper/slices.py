"""Redraw the slice figure (fig_main_text1.5.pdf) with the levels of F/FINAL_LEVELS.json: copies of the cached figure data with
only rho (the orange boundary) and r_d (the distance-baseline circle) replaced, so nothing is evaluated again.
"""
import json, sys
from pathlib import Path
import numpy as np
from paths import REPO, OUT
L = REPO / "lie_rl"; sys.path.insert(0, str(L / "scripts"))
SCR = OUT; (SCR / "figdata").mkdir(exist_ok=True)
import final_figs as FF
FL = json.load(open(L / "results/review/F/FINAL_LEVELS.json"))["loops"]
for fid in ("SO3__td3_SO3_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s1__base"):
    D = dict(np.load(FF.CACHE / f"{fid}.npz")); ev = json.load(open(L / f"results/final/eval/{fid}.json"))["fns"]
    assert float(D["rho"]) == ev["V"]["level"] and abs(float(D["r_d"]) - np.sqrt(max(ev["Vd"]["level"], 0))) < 1e-12
    D["rho"] = np.asarray(FL[fid]["V"]["level"]); D["r_d"] = np.asarray(np.sqrt(max(FL[fid]["Vd"]["level"], 0.0)))
    print(fid, "rho", ev["V"]["level"], "->", float(D["rho"]), "r_d", float(D["r_d"]))
    np.savez(SCR / "figdata" / f"{fid}.npz", **D)
FF.CACHE = SCR / "figdata"
import os, matplotlib.pyplot as plt
ROWH = float(os.environ.get("ROWH", "0"))
if ROWH:
    _sub = plt.subplots
    plt.subplots = lambda *a, **k: _sub(*a, **{**k, "figsize": (k["figsize"][0], 2 * ROWH)})
FF.draw_main(sys.argv[1], *[float(x) for x in sys.argv[2:]])
