"""tab_main, tab_all, tab_pert_all and tab_pert under the two-search rule: every region of V, V_d and V_P comes from F/FINAL_LEVELS.json, the
oracle, ball, spr and fit from the result files."""
import json, glob, os, sys
from decimal import Decimal, ROUND_HALF_UP
from paths import RES, OUT, REF, rows55
R = str(RES); P = str(REF)
REP = str(REF); OUT = str(OUT); MODE = os.environ.get("MODE", "final")
FL = json.load(open(f"{R}/review/F/FINAL_LEVELS.json"))
q = lambda x, d: str(Decimal(f"{x:.9f}").quantize(Decimal(d), rounding=ROUND_HALF_UP))
f1 = lambda x: q(x, "0.1")
def rng(xs):
    a, b = f1(min(xs)), f1(max(xs)); return a if a == b else f"{a}--{b}"
def cells(path):
    return [[c.strip() for c in l.replace("\\\\", "").split("&")] for l in open(path)
            if "&" in l and "toprule" not in l and not l.lstrip().startswith("group")]
missing = set(FL["missing"])
def fnval(i, ev, n):
    """region (percent of mu0), ball fraction (percent), level, rounds string of function n on loop i"""
    e = ev["fns"][n]
    if MODE == "first" or i in missing:
        if MODE != "first": print("WARNING: no second search yet for", i)
        return 100*e["region_mu0"], 100*e["region_ball_fraction"], e["level"], str(len(e["adversarial_rounds"])), ""
    b = FL["loops"][i][n]; second = b.get("rounds_search2_from", b["rounds_search2"])
    return 100*b["region_mu0"], 100*b["region_ball_fraction"], b["level"], f"{b['rounds_first']}+{second}", ("$^{\\dagger}$" if "rounds_search2_from" in b else "")
ID = lambda G, a, s, v: f"{G}__{a}_{G}_ls-1.5__s{s}__{v}"
nom = {}
for p in sorted(glob.glob(f"{R}/final/eval/*__base.json")):
    i = os.path.basename(p)[:-5]; e = json.load(open(p)); o = json.load(open(f"{R}/rx/oracle/{i}.json"))
    G, d, s, _ = i.split("__"); r = dict(id=i, G=G, alg=d.split("_")[0], seed=int(s[1:]), spr=e["spr"], ball=100*e["ball_mu0"], orc=100*o["region_mu0"], fit=100*e["fit_median"])
    for n in ("V", "Vd", "VP"): r[n], r[n + "_ball"], r[n + "_rho"], r[n + "_rounds"], r[n + "_mark"] = fnval(i, e, n)
    nom[i] = r
assert len(nom) == 39
GR = [("SO2", r"$\SO(2)$"), ("SO3", r"$\SO(3)$"), ("SE2", r"$\SE(2)$"), ("SE3", r"$\SE(3)$")]; AL = [("ppo", "PPO"), ("trpo", "TRPO"), ("sac", "SAC"), ("td3", "TD3")]
L = [r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
     r"group & algorithm & seeds & ball & oracle & $V$ & $V_d$ & $V_P$ & $V$/oracle & $V_d$/oracle & $V_P$/oracle & fit \\", r"\midrule"]
for gi, (g, gl) in enumerate(GR):
    if gi: L.append(r"\midrule")
    first = True
    for a, al in AL:
        rs = [r for r in nom.values() if r["G"] == g and r["alg"] == a]
        if not rs: continue
        c = lambda k: rng([r[k] for r in rs]); cr = lambda k: rng([100*r[k]/r["orc"] for r in rs])
        L.append(f"{gl if first else ''} & {al} & {len(rs)} & {c('ball')} & {c('orc')} & {c('V')} & {c('Vd')} & {c('VP')} & {cr('V')} & {cr('Vd')} & {cr('VP')} & {c('fit')} \\\\")
        first = False
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_main.tex", "w").write("\n".join(L) + "\n")
L = [r"\begin{tabular}{@{}lllrrrrrrrrrr@{}}", r"\toprule",
     r"group & algorithm & seed & $\spr(A)$ & oracle & $V$ & $V_d$ & $V_P$ & $V$/oracle & $V$ (ball) & $\rho$ & rounds & fit \\", r"\midrule"]
for gi, (g, gl) in enumerate(GR):
    if gi: L.append(r"\midrule")
    first = True
    for a, al in AL:
        for r in sorted([r for r in nom.values() if r["G"] == g and r["alg"] == a], key=lambda r: r["seed"]):
            L.append(f"{gl if first else ''} & {al} & {r['seed']} & {q(r['spr'], '0.001')} & {f1(r['orc'])} & {f1(r['V'])} & {f1(r['Vd'])} & {f1(r['VP'])} & "
                     f"{(q(r['V']/r['orc'], '0.01') if MODE == 'first' else f1(100*r['V']/r['orc']))} & {f1(r['V_ball'])} & {f1(r['V_rho'])} & {r['V_rounds']}{r['V_mark']} & {f1(r['fit'])} \\\\")
            first = False
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_all.tex", "w").write("\n".join(L) + "\n")
rows = rows55(); pert = []
for r0 in rows:
    i = r0["id"]; e = json.load(open(r0["src"])); o = json.load(open(f"{R}/rx/oracle/{i}.json"))
    r = dict(r0); r["orc"] = 100*o["region_mu0"]; r["spr"] = e["spr"]; r["fit"] = 100*e["fit_median"]
    for n in ("V", "Vd", "VP"): r[n], r[n + "_ball"], r[n + "_rho"], r[n + "_rounds"], r[n + "_mark"] = fnval(i, e, n)
    pert.append(r)
assert len(pert) == 55
def err(v):
    if v[0] == "g": return f"gain {v[1:]}"
    if v[0] == "a": return f"per-axis gains {v[1:]}"
    return f"misalignment {v[1:]}$^\\circ$"
order = ["g0.5", "g0.75", "g1.25", "g1.5", "a1", "a2", "m5", "m10", "m15", "m30", "m40"]
pert.sort(key=lambda r: (r["G"], order.index(r["v"]), r["alg"], r["seed"]))
mark = {"scratch": "", "carry": "$^{c}$", "carry_after_warm": "$^{c}$", "warm": "$^{w}$"}
L = [r"\begin{tabular}{@{}lllrrrrrrrrrrr@{}}", r"\toprule",
     r"group & policy & actuator error & $\spr(A)$ & oracle & $V$ & $V_d$ & $V_P$ & $V$/oracle & $V_d$/oracle & $V_P$/oracle & $\rho$ & rounds & fit \\", r"\midrule"]
prev = None
for r in pert:
    if prev and r["G"] != prev: L.append(r"\midrule")
    g = "" if r["G"] == prev else f"$\\{r['G'][:2]}({r['G'][2]})$"; prev = r["G"]
    L.append(f"{g} & {r['alg']} {r['seed']} & {err(r['v'])} & {q(r['spr'], '0.01')} & {f1(r['orc'])} & {f1(r['V'])}{mark[r['tier']]} & {f1(r['Vd'])} & {f1(r['VP'])} & "
             f"{f1(100*r['V']/r['orc'])} & {f1(100*r['Vd']/r['orc'])} & {f1(100*r['VP']/r['orc'])} & {f1(r['V_rho'])} & {r['V_rounds']}{r['V_mark']} & {f1(r['fit'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_pert_all.tex", "w").write("\n".join(L) + "\n")
grp = [("gain 0.5--1.5", ["g0.5", "g0.75", "g1.25", "g1.5"]), ("per-axis gains", ["a1", "a2"]),
       (r"misalignment 5--15$^\circ$", ["m5", "m10", "m15"]), (r"misalignment 30--40$^\circ$", ["m30", "m40"])]
L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
     r"group & actuator error & loops & oracle & $V$ & $V_d$ & $V_P$ & $V$/oracle & $V_d$/oracle & $V_P$/oracle \\", r"\midrule"]
for G in ("SE2", "SE3"):
    if G == "SE3": L.append(r"\midrule")
    for k, (name, vs) in enumerate(grp):
        rs = [r for r in pert if r["G"] == G and r["v"] in vs]
        g = f"$\\{G[:2]}({G[2]})$" if k == 0 else ""
        L.append(f"{g} & {name} & {len(rs)} & {rng([r['orc'] for r in rs])} & {rng([r['V'] for r in rs])} & {rng([r['Vd'] for r in rs])} & {rng([r['VP'] for r in rs])} & "
                 f"{rng([100*r['V']/r['orc'] for r in rs])} & {rng([100*r['Vd']/r['orc'] for r in rs])} & {rng([100*r['VP']/r['orc'] for r in rs])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_pert.tex", "w").write("\n".join(L) + "\n")
json.dump(dict(nom=list(nom.values()), pert=pert), open(f"{OUT}/rows_{MODE}.json", "w"), indent=1)
if MODE == "first":
    for t, src in (("tab_main", P), ("tab_all", REP), ("tab_pert_all", P), ("tab_pert", P)):
        new, old = cells(f"{OUT}/{t}.tex"), cells(f"{src}/{t}.tex"); assert len(new) == len(old), t
        bad = [(o, n) for n, o in zip(new, old) if n != o]
        print(t, "rows", len(new), "differing", len(bad))
        for o, n in bad[:6]: print("   paper:", o, "\n   built:", n)
print("written", MODE)
