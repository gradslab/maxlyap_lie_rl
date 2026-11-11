"""The reported level of every function on every reported closed loop under the two-search rule: the smaller of the level
of the eval json (first search) and of F/search2 (second search, from that level); for V on SE3 PPO s1 base and s0 m30 also
F/search2_from (second search from the r_b = 0.2 level, whose hits are violations of the r_b = 0.1 conditions). The region and
checks are those of the run that gave the smaller level. Writes results/review/F/FINAL_LEVELS.json.
Also the ablation variants, z^T P_m z and the candidates before refinement (F/search2_x) where present.
"""
import json, glob, os, sys
from paths import RES, OUT, REF, rows55
R = str(RES); F = f"{R}/review/F"
rd = lambda f: [l.strip() for l in open(f) if l.strip()]
loops = {}
for p in sorted(glob.glob(f"{R}/final/eval/*__base.json")): loops[os.path.basename(p)[:-5]] = p
rows = rows55()
for r in rows: loops[r["id"]] = r["src"]
assert len(loops) == 94, len(loops)
out, missing = {}, []
def pick(ev_fn, cands):
    best = dict(level=ev_fn["level"], region_mu0=ev_fn["region_mu0"], region_ball_fraction=ev_fn["region_ball_fraction"], source="eval",
                rounds_first=len(ev_fn["adversarial_rounds"]), checks_pass=True)
    extra = {}
    for tag, x in cands:
        extra[f"rounds_{tag}"] = len(x["adversarial_rounds"]); extra[f"level_{tag}"] = x["level"]
        if not x["checks_pass"]:
            assert x["entry_violations"] > 0 and ev_fn["entry_violations"] == x["entry_violations"] and x["false_inclusions_mu0"] == 0, (tag, x)
            best["checks_pass"] = False; print("  checks fail as in the eval json:", tag, "entry", x["entry_violations"])
        if x["level"] < best["level"]:
            best.update(level=x["level"], region_mu0=x["region_mu0"], region_ball_fraction=x["region_ball_fraction"], source=tag)
    best.update(extra); best["level_first"] = ev_fn["level"]; best["region_first"] = ev_fn["region_mu0"]
    return best
for i, p in loops.items():
    ev = json.load(open(p)); s2p = f"{F}/search2/{i}.json"
    if not os.path.exists(s2p): missing.append(i); continue
    s2 = json.load(open(s2p)); assert s2["eval"].endswith(os.path.relpath(p, os.path.dirname(R))) or os.path.samefile(os.path.join(os.path.dirname(R), s2["eval"]), p), (i, s2["eval"])
    o = {}
    for n in ("V", "Vd", "VP"):
        if n not in ev["fns"]: continue
        assert abs(s2["fns"][n]["level_reported"] - ev["fns"][n]["level"]) == 0, (i, n)
        c = [("search2", s2["fns"][n])]
        sf = f"{F}/search2_from/{i}.json"
        if n == "V" and os.path.exists(sf): c.append(("search2_from", json.load(open(sf))["fns"]["V"]))
        o[n] = pick(ev["fns"][n], c)
    out[i] = o
x = {}
for p in sorted(glob.glob(f"{F}/search2_x/*.json")):
    k = os.path.basename(p)[:-5]; i, tag = k.rsplit("__", 1); s = json.load(open(p))
    ev = json.load(open(f"{R}/{s['eval'].split('results/', 1)[1]}")); n = "VPm" if tag == "VPm" else "V"
    x[k] = pick(ev["fns"][n], [("search2", s["fns"][n])]); x[k]["known_points"] = s["fns"][n].get("known_points")
json.dump(dict(loops=out, extra=x, missing=missing), open(f"{F}/FINAL_LEVELS.json", "w"), indent=1)
print("loops", len(out), "missing", len(missing), "extra", len(x))
for i, o in sorted(out.items()):
    for n, b in o.items():
        if b["source"] != "eval": print(f"  {i} {n}: {100*b['region_first']:.3f} -> {100*b['region_mu0']:.3f} ({b['source']})")
