"""Figure 1 panels (a) and (b): composition of the two 3D renders of make_fig1_v4.py."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import make_fig1_v4 as V4
M = V4.M
FL = json.load(open(M.RES / "review" / "F" / "FINAL_LEVELS.json"))["loops"]
for n in ("V", "VP"):
    assert FL[M.FID][n]["source"] == "eval", (M.FID, n)

def loops55_pb():
    """(id, V, Vd, VP, oracle) in percent for the 55 closed loops with an actuator error, under the two-search rule."""
    out = []
    for r in M.loops55():
        i = r[0]; f = FL[i]
        out.append((i, 100 * f["V"]["region_mu0"], 100 * f["Vd"]["region_mu0"], 100 * f["VP"]["region_mu0"], r[3]))
    v, d_, p_, o = (np.array([x[k] for x in out]) for k in (1, 2, 3, 4))
    print("margins over Vd %.4f..%.4f median %.4f; over VP %.4f..%.4f median %.4f; medians V/Vd/VP over oracle %.4f %.4f %.4f"
          % ((v - d_).min(), (v - d_).max(), np.median(v - d_), (v - p_).min(), (v - p_).max(), np.median(v - p_),
             np.median(100 * v / o), np.median(100 * d_ / o), np.median(100 * p_ / o)))
    assert (v > d_).all() and (v > p_).all()
    return out

V4.loops55_pb = loops55_pb
if __name__ == "__main__":
    assert sys.argv[1] == "compose"
    D = dict(np.load(M.CACHE)); stem = sys.argv[2] if len(sys.argv) > 2 else "fig1_v5"
    V4.compose(D, V4.load_img(V4.SCR / "panel_a.png"), V4.load_img(V4.SCR / "panel_b.png"), str(V4.SCR / stem))
