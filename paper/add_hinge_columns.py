"""Add the hinge-only baseline V_h to tab_pert.tex (grouped ranges) and tab_pert_all.tex (one row per loop) of the paper folder,
from hb_levels.json and the oracle files. V_h and V_h/oracle are inserted after V_P and V_P/oracle. A star marks a loop on which
V_h gives no region (region 0, entry check fails). Prints the summary numbers used in the text.
"""
import json, re, sys, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from paths import RES, OUT
O = sys.argv[1] if len(sys.argv) > 1 else str(OUT); R = str(RES)
H = {r["id"]: r for r in json.load(open(f"{R}/review/hinge_base/hb_levels.json"))["comparison"]}
orc = {i: 100 * json.load(open(f"{R}/rx/oracle/{i}.json"))["region_mu0"] for i in H}
q = lambda x, d="0.1": str(Decimal(f"{x:.9f}").quantize(Decimal(d), rounding=ROUND_HALF_UP))
def rng(xs):
    a, b = q(min(xs)), q(max(xs)); return a if a == b else f"{a}--{b}"
def var(err):
    err = err.replace("$^\\circ$", "").strip()
    if err.startswith("gain"): return "g" + err.split()[1]
    if err.startswith("per-axis"): return "a" + err.split()[2]
    return "m" + err.split()[1]
G_of = lambda c: c.replace("$\\", "").replace("(", "").replace(")$", "").strip()
L = open(f"{O}/tab_pert_all.tex").read().splitlines(); out = []; G = None; n = 0
for l in L:
    if l.startswith("\\begin{tabular}"): out.append(l.replace("lllrrrrrrrrrrr", "lllrrrrrrrrrrrrr")); continue
    if "&" not in l: out.append(l); continue
    c = [x.strip() for x in l.replace("\\\\", "").split("&")]
    if c[0] == "group":
        c = c[:8] + ["$V_h$"] + c[8:11] + ["$V_h$/oracle"] + c[11:]; out.append(" & ".join(c) + " \\\\"); continue
    if c[0]: G = G_of(c[0])
    alg, seed = c[1].split(); i = f"{G}__{alg.lower()}_{G}_ls-1.5__s{seed}__{var(c[2])}"
    h = 100 * H[i]["hinge"]; assert q(orc[i]) == c[4], (i, q(orc[i]), c[4])
    hv = q(h) + ("$^{*}$" if h == 0 else ""); c = c[:8] + [hv] + c[8:11] + [q(100 * h / orc[i])] + c[11:]; out.append(" & ".join(c) + " \\\\"); n += 1
assert n == 55, n
open(f"{O}/tab_pert_all.tex", "w").write("\n".join(out) + "\n")
classes = {"gain 0.5--1.5": ("g",), "per-axis gains": ("a",), "misalignment 5--15$^\\circ$": ("m5", "m10", "m15"), "misalignment 30--40$^\\circ$": ("m30", "m40")}
def members(G, name):
    keys = classes[name]
    return [i for i in H if i.startswith(G + "__") and (any(i.endswith("__" + k) for k in keys) if keys[0] != "g" and keys[0] != "a" else i.split("__")[-1].startswith(keys[0]))]
L = open(f"{O}/tab_pert.tex").read().splitlines(); out = []; G = None
for l in L:
    if l.startswith("\\begin{tabular}"): out.append(l.replace("llrrrrrrrr", "llrrrrrrrrrr")); continue
    if "&" not in l: out.append(l); continue
    c = [x.strip() for x in l.replace("\\\\", "").split("&")]
    if c[0] == "group":
        c = c[:7] + ["$V_h$"] + c[7:10] + ["$V_h$/oracle"]; out.append(" & ".join(c) + " \\\\"); continue
    if c[0]: G = G_of(c[0])
    ids = members(G, c[1]); assert len(ids) == int(c[2]), (G, c[1], len(ids), c[2])
    hs = [100 * H[i]["hinge"] for i in ids]; hr = [100 * h / orc[i] for h, i in zip(hs, ids)]
    c = c[:7] + [rng(hs)] + c[7:10] + [rng(hr)]; out.append(" & ".join(c) + " \\\\")
open(f"{O}/tab_pert.tex", "w").write("\n".join(out) + "\n")
ratio = [100 * 100 * H[i]["hinge"] / orc[i] for i in H]; rV = [100 * 100 * H[i]["V"] / orc[i] for i in H]; rd = [100 * 100 * H[i]["Vd"] / orc[i] for i in H]; rp = [100 * 100 * H[i]["VP"] / orc[i] for i in H]
print("median V_h/oracle %.1f (V %.1f, Vd %.1f, VP %.1f)" % (st.median(ratio), st.median(rV), st.median(rd), st.median(rp)))
print("V_h/oracle >= 99 on", sum(x >= 99 for x in ratio), "; V_h = 0 on", sum(H[i]["hinge"] == 0 for i in H))
gain = [100 * (H[i]["V"] - H[i]["hinge"]) for i in H]; print("V - V_h median %.1f min %.1f max %.1f" % (st.median(gain), min(gain), max(gain)))
print("tables written")
