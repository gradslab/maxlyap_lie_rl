"""tab_ablation and the numbers of the ablation paragraphs (B2: z^T P_m z alone, B3: refinement, B9: variants) under the
two-search rule. MODE=first rebuilds from the first search only and checks the table against the paper.
"""
import json, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from paths import RES, OUT as _OUT, REF
R = str(RES); P = str(REF)
OUT = str(_OUT); MODE = os.environ.get("MODE", "final")
FL = json.load(open(f"{R}/review/F/FINAL_LEVELS.json"))
f1 = lambda x: str(Decimal(f"{x:.9f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
ids = [l.strip() for l in open(f"{R}/review/nontrivial_ids.txt") if l.strip() and l.split("__")[3].strip() not in ("m50", "m60")]
def evp(i):
    for p in (f"{R}/final/eval/{i}.json", f"{R}/review/pert36/eval/{i}__final.json"):
        if os.path.exists(p): return p
    raise FileNotFoundError(i)
def cand(i):
    if MODE == "first": return 100*json.load(open(evp(i)))["fns"]["V"]["region_mu0"]
    return 100*FL["loops"][i]["V"]["region_mu0"]
def var(i, tag):
    n = "VPm" if tag == "VPm" else "V"
    ev = f"{R}/review/eval_VPm/{i}.json" if tag == "VPm" else f"{R}/review/ablation/eval/{i}__{tag}.json"
    first = 100*json.load(open(ev))["fns"][n]["region_mu0"]
    if MODE == "first": return first
    k = f"{i}__{tag}"
    if k not in FL["extra"]: print("WARNING: no second search yet for", k); return first
    return 100*FL["extra"][k]["region_mu0"]
orc = lambda i: 100*json.load(open(f"{R}/rx/oracle/{i}.json"))["region_mu0"]
abl = [("SO3__td3_SO3_ls-1.5__s1__base", r"$\SO(3)$", "TD3 1", "none"), ("SE2__ppo_SE2_ls-1.5__s1__base", r"$\SE(2)$", "PPO 1", "none"),
       ("SE3__ppo_SE3_ls-1.5__s1__base", r"$\SE(3)$", "PPO 1", "none"), ("SE3__ppo_SE3_ls-1.5__s2__base", r"$\SE(3)$", "PPO 2", "none"),
       ("SE3__ppo_SE3_ls-1.5__s1__g1.5", r"$\SE(3)$", "PPO 1", "gain 1.5")]
L = [r"\begin{tabular}{@{}lllrrrrrr@{}}", r"\toprule",
     r"group & policy & actuator error & candidate & no quadratic term & hinge only & regression & no refinement & oracle \\", r"\midrule"]
loss = {t: [] for t in ("noanchor", "hinge", "regress", "norefine")}
for i, g, pol, er in abl:
    c = cand(i); v = {t: var(i, t) for t in loss}
    for t in loss: loss[t].append((i, c - v[t]))
    star = "$^{*}$" if i == "SE3__ppo_SE3_ls-1.5__s2__base" else ""
    L.append(f"{g} & {pol} & {er} & {f1(c)} & {f1(v['noanchor'])} & {f1(v['hinge'])} & {f1(v['regress'])}{star} & {f1(v['norefine'])} & {f1(orc(i))} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_ablation.tex", "w").write("\n".join(L) + "\n")
if MODE == "first":
    new = [l for l in open(f"{OUT}/tab_ablation.tex")]; old = [l for l in open(f"{P}/tab_ablation.tex")]
    print("tab_ablation identical to the paper:", new == old)
    if new != old:
        for a, b in zip(old, new):
            if a != b: print("  paper:", a.strip(), "\n  built:", b.strip())
print("== B9 variants (candidate minus variant, points)")
for t, xs in loss.items(): print(f"  {t}: " + ", ".join(f"{i.split('__')[0]} {i.split('__')[1].split('_')[0]} {i.split('__')[2]} {i.split('__')[3]}: {x:.3f}" for i, x in xs))
print("== B2 z^T P_m z alone over the 16 (candidate minus VPm)")
b2 = sorted(((i, cand(i) - var(i, "VPm")) for i in ids), key=lambda t: t[1])
for i, x in b2: print(f"  {i}: {x:.3f}")
big = [x for _, x in b2 if x > 1]; print(f"  smaller by >1 point on {len(big)}: {f1(min(big))} .. {f1(max(big))}; the rest within {max(abs(x) for _, x in b2 if x <= 1):.3f}")
print("== B3 refinement over the 16 (candidate minus no refinement)")
b3 = sorted(((i, cand(i) - var(i, "norefine")) for i in ids), key=lambda t: t[1])
for i, x in b3: print(f"  {i}: {x:.3f}")
gt = [x for _, x in b3 if abs(x) > 1]; print(f"  changes by >1 point on {len(gt)}: {[round(x, 2) for x in gt]}; most negative {min(x for _, x in b3):.4f}")
