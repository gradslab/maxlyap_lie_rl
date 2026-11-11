"""Hinge-only baseline on the 55 actuator-error loops: reported level = min(first run of the level rule, second search),
region and checks of the run that gave the smaller level (as final_levels.py does for V, Vd, VP). Writes hb_levels.json
and prints the comparison with V, Vd, VP and the oracle from F/FINAL_LEVELS.json (paths relative to lie_rl/).
"""
import json, glob, os, statistics as st
from paths import RES
R = f"{RES}/review"; L = json.load(open(f"{R}/F/FINAL_LEVELS.json"))["loops"]
rows = {}
for f in sorted(glob.glob(f"{R}/hinge_base/search2/*.json")):
    i = os.path.basename(f)[:-len("__hinge.json")]
    s2 = json.load(open(f)); ev = json.load(open(f"{R}/hinge_base/eval/{i}__hinge.json"))
    a, b = ev["fns"]["V"], s2["fns"]["V"]
    assert abs(b["level_reported"] - a["level"]) == 0, i
    best = dict(level=a["level"], region_mu0=a["region_mu0"], source="eval", rounds_first=len(a["adversarial_rounds"]),
                rounds_search2=len(b["adversarial_rounds"]), level_first=a["level"], region_first=a["region_mu0"], level_search2=b["level"])
    if b["level"] < a["level"]:
        best.update(level=b["level"], region_mu0=b["region_mu0"], source="search2")
    chk = dict(entry_violations=b["entry_violations"], nb_nonconverged=b["nb_nonconverged"], false_inclusions_mu0=b["false_inclusions_mu0"],
               false_inclusions_ball=b["false_inclusions_ball"], checks_pass=b["checks_pass"], entry_samples=b["entry_samples"])
    best.update(chk); best["oracle_mu0"] = ev.get("true_in_ball_mu0")
    rows[i] = best
assert len(rows) == 55, len(rows)
def parts(i): g, d, s, v = i.split("__"); return g, d.split("_")[0], int(s[1:]), v
out = []
for i, h in rows.items():
    V, Vd, VP = (L[i][k]["region_mu0"] for k in ("V", "Vd", "VP"))
    out.append(dict(id=i, group=parts(i)[0], alg=parts(i)[1], seed=parts(i)[2], err=parts(i)[3], hinge=h["region_mu0"], V=V, Vd=Vd, VP=VP,
                    hinge_source=h["source"], entry_violations=h["entry_violations"], nb_nonconverged=h["nb_nonconverged"],
                    false_inclusions_mu0=h["false_inclusions_mu0"], checks_pass=h["checks_pass"]))
json.dump(dict(levels=rows, comparison=out), open(f"{R}/hinge_base/hb_levels.json", "w"), indent=1)
pp = lambda x: f"{100*x:.1f}"
gain = [100 * (r["V"] - r["hinge"]) for r in out]
print("loops", len(out), "| checks pass on", sum(r["checks_pass"] for r in out), "| entry violations > 0 on", sum(r["entry_violations"] > 0 for r in out),
      "| nb nonconverged > 0 on", sum(r["nb_nonconverged"] > 0 for r in out), "| false inclusions > 0 on", sum(r["false_inclusions_mu0"] > 0 for r in out))
print("second search lowered the hinge level on", sum(r["hinge_source"] == "search2" for r in out))
print("V > hinge by > 0.01 pts on", sum(g > 0.01 for g in gain), "; |V - hinge| <= 0.01 on", sum(abs(g) <= 0.01 for g in gain), "; hinge > V by > 0.01 on", sum(g < -0.01 for g in gain))
print("V - hinge (pts): median %.1f, min %.1f, max %.1f" % (st.median(gain), min(gain), max(gain)))
print("hinge > max(Vd,VP) by > 0.01 on", sum(100 * (r["hinge"] - max(r["Vd"], r["VP"])) > 0.01 for r in out), "; hinge < max(Vd,VP) by > 0.01 on", sum(100 * (r["hinge"] - max(r["Vd"], r["VP"])) < -0.01 for r in out))
hd = [100 * (r["hinge"] - r["Vd"]) for r in out]; hp = [100 * (r["hinge"] - r["VP"]) for r in out]
print("hinge - Vd: median %.1f (min %.1f, max %.1f); hinge - VP: median %.1f (min %.1f, max %.1f)" % (st.median(hd), min(hd), max(hd), st.median(hp), min(hp), max(hp)))
print("hinge region == 0 on", sum(r["hinge"] == 0 for r in out))
print()
print(f"{'loop':34s} {'hinge':>6s} {'V':>6s} {'Vd':>6s} {'VP':>6s}  V-h   src  entry nbnc fi")
for r in sorted(out, key=lambda r: (r["group"], r["alg"], r["seed"], r["err"])):
    print(f"{r['id']:34s} {pp(r['hinge']):>6s} {pp(r['V']):>6s} {pp(r['Vd']):>6s} {pp(r['VP']):>6s} {100*(r['V']-r['hinge']):5.1f}  {r['hinge_source'][:6]:6s} {r['entry_violations']:5d} {r['nb_nonconverged']:4d} {r['false_inclusions_mu0']:2d}")
