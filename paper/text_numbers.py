"""Every number of Sections 5 and 6 and of the appendices that no other script in this folder prints, recomputed from the
result files and compared with the value printed in the paper. Run final_levels.py and aggregate_hb.py first; derived.json and
rloc_summary_SO2.json in results/review/F come from derive.py and rloc_summary.py. Prints one line per group of numbers and
exits with status 1 if any group differs. Rounding is half up, as in build2.py.
"""
import glob
import json
import math
import os
import pickle
import re
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from paths import RES, rows55

R = str(RES)
_cache = {}
results = []


def J(p):
    p = str(p)
    if p not in _cache:
        _cache[p] = json.load(open(p if os.path.isabs(p) else f"{R}/{p}"))
    return _cache[p]


def PK(p):
    p = str(p)
    if p not in _cache:
        _cache[p] = pickle.load(open(p if os.path.isabs(p) else f"{R}/{p}", "rb"))
    return _cache[p]


def q(x, d="0.1"):
    return str(Decimal(f"{x:.9f}").quantize(Decimal(d), rounding=ROUND_HALF_UP))


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}e{int(e)}"


def rng(xs, d="0.1", d2=None):
    return f"{q(min(xs), d)} to {q(max(xs), d2 or d)}"


def check(name, got, want):
    got = [str(x) for x in (got if isinstance(got, (list, tuple)) else [got])]
    want = [str(x) for x in (want if isinstance(want, (list, tuple)) else [want])]
    results.append((got == want, name, "; ".join(got), "; ".join(want)))


def bound(name, x, op, lim, fmt=str):
    ok = dict(lt=x < lim, le=x <= lim, gt=x > lim)[op]
    results.append((ok, name, fmt(x), dict(lt="below ", le="at most ", gt="above ")[op] + fmt(lim)))


def about(name, x, want, tol=0.05, fmt=lambda v: f"{v:.3g}"):
    results.append((abs(x - want) <= tol * abs(want), name, f"{fmt(x)} (about)", f"about {want:g}"))


def parts(i):
    G, d, s, v = i.split("__")
    return G, d.split("_")[0], int(s[1:]), v


def nice(i):
    G, a, s, v = parts(i)
    return f"{G} {a.upper()} s{s} {v}"


def rel(p):
    return R + "/" + p.split("results/", 1)[1]


ROWS = rows55()
I55 = [r["id"] for r in ROWS]
TIER = {r["id"]: r["tier"] for r in ROWS}
SRC = {r["id"]: r["src"] for r in ROWS}
I39 = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{R}/final/eval/*__base.json"))
I94 = I39 + I55
FLJ = J("review/F/FINAL_LEVELS.json")
FL, FX = FLJ["loops"], FLJ["extra"]
SCR55 = [i for i in I55 if TIER[i] == "scratch"]
WARM = [i for i in I55 if TIER[i] == "warm"]
CARRY = [i for i in I55 if TIER[i].startswith("carry")]
LATER = WARM + CARRY
SCRATCH65 = I39 + SCR55
SE = ("SE2", "SE3")


def EV(i):
    return J(f"final/eval/{i}.json") if i.endswith("__base") else J(SRC[i])


def S2(i):
    return J(f"review/F/search2/{i}.json")


def extra(i):
    for d in ("review/extra", "review/pert36/extra", "review/pert_r2/extra", "review/pert_r3/extra"):
        if os.path.exists(f"{R}/{d}/{i}.json"):
            return J(f"{d}/{i}.json")
    raise FileNotFoundError(i)


def nominal(i):
    p = f"review/nominal_T/{i}.json"
    return J(p) if os.path.exists(f"{R}/{p}") else J(f"review/nominal/{i}.json")


def kappa(c):
    return c["cfg"]["alpha_floor"] + math.log1p(math.exp(c["alpha_raw"]))


def orc(i):
    return 100 * J(f"rx/oracle/{i}.json")["region_mu0"]


def so2_V(i):
    """The candidate on SO(2) evaluated in numpy from its stored parameters."""
    c = PK(rel(EV(i)["cert"]))
    W = [(np.asarray(w, float), np.asarray(b, float)) for w, b in c["psi"]]

    def mlp(x):
        for w, b in W[:-1]:
            x = np.maximum(x @ w + b, 0)
        return x @ W[-1][0] + W[-1][1]
    k, P, ra = kappa(c), float(np.asarray(c["anchor"]["P"])[0, 0]), c["anchor"]["r_a"]
    f0 = mlp(np.zeros((1, 1)))

    def V(z):
        f = mlp(z[:, None]) - f0
        r2 = z * z
        s = r2 / (r2 + ra * ra)
        return (1 - s) * P * r2 + s * ((f * f).sum(1) + k * k * 4 * (1 - np.cos(z)))
    return V


def spr(M):
    return np.abs(np.linalg.eigvals(M)).max(axis=-1)


def s_policies():
    """39 of 48 policies pass the stability test; the nine that fail; final errors and offsets of the failing ones."""
    tags = dict(ppo="", trpo="_trpo", sac="_sac", td3="_td3")
    rho, off = {}, {}
    for G in ("SO2", "SO3", "SE2", "SE3"):
        for a, t in tags.items():
            for s in range(3):
                d = J(f"l4dc/roa_{G}{t}_seed{s}.json")
                rho[(G, a, s)] = d["fixed_points"]["rho_goal"]
                off[(G, a, s)] = d["offset_norm"]
    fail = [k for k, v in rho.items() if v > 0.99]
    passing = sorted(f"{G}__{a}_{G}_ls-1.5__s{s}__base" for (G, a, s), v in rho.items() if v <= 0.99)
    six = [k for k in fail if k[0] == "SE3" and k[1] in ("sac", "td3")]
    check("policies passing the stability test, of 48; failing; failing SAC/TD3 on SE(3); passing set = evaluated set",
          [len(rho) - len(fail), len(rho), len(fail), len(six), passing == I39], [39, 48, 9, 6, True])
    pe = J("l4dc/policy_eval_common.json")["SE3"]
    errs = [pe[f"{a}_{s}"][k] for (_, a, s) in six for k in ("rot_median", "trans_median")]
    check("median final errors of the six SAC/TD3 policies on SE(3), within 0.06 to 0.18", [min(errs) >= 0.06, max(errs) <= 0.18], [True, True])
    check("offsets of the TD3 policies failing on the rotation groups", [q(off[k], "0.01") for k in (("SO2", "td3", 0), ("SO3", "td3", 0), ("SO3", "td3", 2))],
          ["1.00", "1.43", "1.73"])
    am = {G: J(f"l4dc/roa_{G}_seed0.json")["alpha_max"] for G in ("SO2", "SO3", "SE2", "SE3")}
    check("clip length alpha_max on rotation and rigid-motion groups", [q(am["SO2"], "0.001"), q(am["SO3"], "0.001"), q(am["SE2"], "0.001"), q(am["SE3"], "0.001")],
          ["0.314", "0.314", "0.351", "0.351"])
    check("closed loops evaluated: 39 policies plus 55 with an actuator error", [len(I39), len(I55), len(set(I94)), len(FL)], [39, 55, 94, 94])


def s_ball():
    """Ball fractions of mu_0 and convergence of the initial errors in the ball."""
    b = {G: {round(100 * EV(i)["ball_mu0"], 9) for i in I39 if i.startswith(G)} for G in ("SO2", "SO3", "SE2", "SE3")}
    check("ball holds percent of the mu_0 samples (SO2, SO3, SE2, SE3)", [q(next(iter(b[G]))) for G in b] + [all(len(v) == 1 for v in b.values())],
          ["87.6", "75.4", "82.0", "65.1", True])
    check("ball on SO(2) holds exactly gamma/pi of mu_0", q(100 * (np.pi - np.pi / 8) / np.pi, "0.1"), "87.5")
    ok = [i for i in I94 if EV(i)["true_in_ball_mu0"] == EV(i)["ball_mu0"]]
    check("every sampled initial error in the ball converges: policies, closed loops", [sum(i in ok for i in I39), len(ok)], [39, 94])
    t = {i: EV(i)["true_mu0"] for i in I39}
    off = sorted(i for i in I39 if t[i] < 1)
    check("every mu_0 initial error converges on 37 of 39; PPO s0 and s1 on SE(3)",
          [sum(v == 1 for v in t.values())] + [nice(i) + " " + q(100 * t[i], "0.001") for i in off],
          [37, "SE3 PPO s0 base 99.978", "SE3 PPO s1 base 99.987"])
    i = "SE3__ppo_SE3_ls-1.5__s0__m40"
    check("PPO s0 SE(3) misalignment 40: oracle region, ball, all in-ball samples converge",
          [q(orc(i)), q(100 * EV(i)["ball_mu0"]), EV(i)["true_in_ball_mu0"] == EV(i)["ball_mu0"]], ["63.5", "65.1", True])
    tb = {i: EV(i)["true_ball_fraction"] for i in I94}
    rest = [i for i in I94 if tb[i] < 1]
    kinds = all(parts(i)[0] in SE and (parts(i)[1], parts(i)[2]) in (("ppo", 1), ("ppo", 2), ("trpo", 1)) for i in rest)
    check("uniform ball sample converges on 69; others 25, all PPO s1/s2 or TRPO s1 on SE(n), percent failing",
          [len(I94) - len(rest), len(rest), kinds, f"{q(100 * (1 - max(tb[i] for i in rest)), '0.001')} to {q(100 * (1 - min(tb[i] for i in rest)), '0.01')}"],
          [69, 25, True, "0.002 to 0.64"])


def s_levels_fit():
    """Largest reported level and the fit of the candidates."""
    lv = max(FL[i]["V"]["level"] for i in I94)
    check("largest reported level of V", q(lv), "265.5")
    fit = [100 * EV(i)["fit_median"] for i in SCRATCH65 + WARM]
    check("fit of the candidates trained on their closed loop (scratch and refitted, 80), min to max", [len(fit), rng(fit, "0.01", "0.1")], [80, "0.05 to 6.7"])
    check("fit of the 14 kept candidates", rng([100 * EV(i)["fit_median"] for i in CARRY]), "2.6 to 45.4")
    same = [i for i in I94 if extra(i)["fit"]["600"]["median"] == extra(i)["fit"]["3000"]["median"]]
    check("fit identical with 600 and 3000 simulated steps", len(same), 94)


def s_heldout():
    """Held-out samples: 10^6 uniform points per closed loop at the level of the first search run."""
    bad, ok, lvl, nn = [], 0, True, set()
    for i in I94:
        x = extra(i)
        h = x["heldout"]
        nn.add(h["n"])
        lvl &= x["level"] == FL[i]["V"]["level_first"]
        if h["decrease_violations"] + h["containment_violations"] == 0:
            ok += 1
        else:
            bad.append((nice(i), h["decrease_violations"], h["containment_violations"]))
    check("held-out points per loop; loops with no failing point; at the first-run level", [sorted(nn), ok, lvl], [[1000000], 93, True])
    check("the exception: TRPO s2 on SO(2), decrease and containment failures", bad, [("SO2 TRPO s2 base", 0, 1)])
    check("bound ln(10^6)/10^6 with confidence 1 - 10^-6", [sci(math.log(1e6) / 1e6), sci(extra(I39[0])["heldout"]["bound_ball_fraction"])], ["1.4e-5", "1.4e-5"])
    g = J("review/grid_so2/SO2__trpo_SO2_ls-1.5__s2__base.json")
    gam = np.pi - np.pi / 8
    others = [J(f) for f in glob.glob(f"{R}/review/grid_so2/*.json") if "trpo_SO2_ls-1.5__s2" not in f]
    check("SO(2) grid: points, spacing, failing points, distance to the rim, level factor, region unchanged, other ten clean",
          [sci(g["n_grid"], 0), sci(g["spacing"]), g["containment_violations"] + g["decrease_violations"], sci(max(gam - abs(z) for z in g["violating_z"])),
           sci(1 - g["lowered_level"] / g["level"]), g["region_mu0_lowered"] == g["region_mu0"],
           len(others), all(o["containment_violations"] + o["decrease_violations"] == 0 for o in others)],
          ["2e7", "2.7e-7", 5, "1.1e-6", "1.1e-6", True, 10, True])


def s_so2proof():
    """Interval proof on SO(2)."""
    D = {os.path.basename(f)[:-5]: J(f) for f in sorted(glob.glob(f"{R}/review/F/so2cert_v3/*.json"))}
    t = "SO2__trpo_SO2_ls-1.5__s2__base"
    ten = [k for k in D if k != t]
    cell = D[t]["unproved_at_reported_level"]["containment"]
    sac = sorted(k for k in D if not D[k]["local_contraction_proved"])
    check("SO(2): proved on, at the reported level on, cells, bisections at most",
          [sum(d["proved"] for d in D.values()), sum(d["proved_at_reported_level"] for d in D.values()), {d["depth_history"][0]["cells"] for d in D.values()} == {2 ** 14},
           max(len(d["depth_history"]) - 1 for d in D.values()) <= 16], [11, 10, True, True])
    check("SO(2): contraction on N_b on, the others are the three SAC loops proved by |F(z)| < |z|",
          [11 - len(sac), [parts(k)[1] for k in sac], all(D[k]["local_ratio_proved"] for k in sac)], [8, ["sac", "sac", "sac"], True])
    check("SO(2) TRPO s2: failing cell width, relative level drop, at the edge of the ball",
          [len(cell), sci(cell[0][1] - cell[0][0]), sci(1 - D[t]["level_proved_over_reported"]), cell[0][1] == D[t]["gamma"]], [1, "1.4e-6", "1.2e-6", True])
    check("SO(2): proved regions (ten, TRPO s2) and sampled sizes",
          [{q(100 * D[k]["region_mu0_proved_lower"], "0.01") for k in ten} == {"87.50"}, q(100 * D[t]["region_mu0_proved_lower"], "0.01"),
           {q(100 * EV(k)["fns"]["V"]["region_mu0"], "0.01") for k in ten} == {"87.56"}, q(100 * EV(t)["fns"]["V"]["region_mu0"], "0.01")],
          [True, "86.03", True, "86.14"])
    check("SO(2): smallest proved decrease outside N_b", rng([d["min_decrease_margin"] for d in D.values()], "0.0001"), "0.0099 to 0.0170")
    z = np.linspace(-D[t]["gamma"], D[t]["gamma"], 40001)
    V = so2_V(t)(z)
    rho = D[t]["level_reported"]
    g = J(f"review/grid_so2/{t}.json")
    run = np.flatnonzero(V > rho)
    check("SO(2) TRPO s2: candidate recomputed (fraction of the grid in the region agrees with the 2e7-point grid to 1e-4); V exceeds rho on one interval",
          [abs((V <= rho).mean() - g["n_in_region"] / g["n_grid"]) < 1e-4, bool(np.all(np.diff(run) == 1)), q(z[run].min(), "0.01"), q(z[run].max(), "0.01")],
          [True, True, "2.66", "2.75"])


def s_fixed_point():
    """Second fixed point of TRPO seed 2 on SO(2) and the baseline regions that reach it."""
    dv = J("review/F/derived.json")
    zs = dv["fixed_point"]["z"]
    gam = np.pi - np.pi / 8
    i = "SO2__trpo_SO2_ls-1.5__s2__base"
    a = np.array(dv["A0"]["SO2__trpo_SO2_ls-1.5__s2"]["A"])[0, 0]
    rd = math.sqrt(FL[i]["Vd"]["level"])
    rp = math.sqrt(FL[i]["VP"]["level"] * (1 - a * a))
    z = np.linspace(zs - 0.03, zs + 0.03, 6001)
    stop = bool(np.all(so2_V(i)(z) > FL[i]["V"]["level"]))
    check("second fixed point inside the ball (z*, gamma), repelling, candidate region stops before it",
          [q(zs, "0.001"), q(gam, "0.001"), dv["fixed_point"]["f_left"] < 0 < dv["fixed_point"]["f_right"], stop],
          ["2.696", "2.749", True, True])
    check("V_d and V_P regions extend past z* (error norm), same set", [sci(rd - zs), sci(rp - zs), abs(rd - rp) < 1e-9], ["8.4e-6", "8.4e-6", True])
    about("mu_0-measure of that excess", (rd - zs) / (2 * np.pi), 1.3e-6)


def s_nominal_td3():
    """TD3 seed 1 on SO(3): baselines and the quadratic term below 0.01 percent; V_P entry check."""
    i = "SO3__td3_SO3_ls-1.5__s1__base"
    vals = [100 * FL[i]["Vd"]["region_mu0"], 100 * FL[i]["VP"]["region_mu0"], 100 * FX[i + "__VPm"]["region_mu0"]]
    vp = EV(i)["fns"]["VP"]
    bound("TD3 s1 SO(3): largest of the V_d, V_P and z'P_m z regions, percent", max(vals), "lt", 0.01, sci)
    check("TD3 s1 SO(3): V gives the whole ball", abs(FL[i]["V"]["region_mu0"] - EV(i)["ball_mu0"]) <= 1e-4, True)
    check("TD3 s1 SO(3): V_P entry fails at sampled points of N_b; its region holds sampled initial errors",
          [vp["entry_violations"], vp["entry_samples"], round(FL[i]["VP"]["region_mu0"] * 2e5)], [93, 39145, 1])


def s_screens():
    """The three screens, the gaps and the selected closed loops."""
    lists = {}
    for k, f in enumerate(("screen_jobs.txt", "screen_jobs_r2.txt", "screen_jobs_r3.txt")):
        for l in open(f"{R}/rx/{f}"):
            G, d, s, v = l.split()
            if v.startswith("m") and float(v[1:]) > 40:
                continue
            lists[f"{G}__{d}__s{s}__{v}"] = k
    S = {i: J(f"rx/screen/{i}.json") for i in lists}
    gap = {i: 100 * (x["true_in_ball_mu0"] - max(x["dist_mu0"], x["quad_mu0"])) for i, x in S.items()}
    big = sorted(i for i in S if parts(i)[0] in SE and gap[i] >= 5)
    so = [gap[i] for i in S if parts(i)[0] not in SE]
    check("screened closed loops: total, per screen", [len(S), [sum(v == k for v in lists.values()) for k in range(3)]], [300, [102, 132, 66]])
    check("screens: every closed loop passes the stability test", all(x["passes_stability"] for x in S.values()), True)
    bound("screens: largest spr", max(x["spr"] for x in S.values()), "lt", 0.80, lambda v: q(v, "0.001"))
    bound("screens: largest spr with per-axis gains", max(x["spr"] for i, x in S.items() if parts(i)[3][0] == "a"), "lt", 0.61, lambda v: q(v, "0.001"))
    bound("screens: largest gap on SO(2) and SO(3), points", max(so), "lt", 1.7, lambda v: q(v, "0.001"))
    check("screens: SE(n) loops, gap >= 5 points, per screen, the other SE(n) loops, = the 55 evaluated",
          [sum(parts(i)[0] in SE for i in S), len(big), [sum(lists[i] == k for i in big) for k in range(3)],
           sum(parts(i)[0] in SE for i in S) - len(big), big == sorted(I55)], [198, 55, [26, 25, 4], 143, True])
    lead = {}
    for i in I39:
        if parts(i)[0] in SE:
            lead[i.rsplit("__", 1)[0]] = 100 * (FL[i]["V"]["region_mu0"] - max(FL[i]["Vd"]["region_mu0"], FL[i]["VP"]["region_mu0"]))
    five = sorted(k for k, v in lead.items() if v >= 6.0 - 5e-4)
    own = [i for i in I55 if i.rsplit("__", 1)[0] in five]
    rest = [i for i in I55 if i not in own]
    check("of the 55, loops of the five SE(n) policies whose candidate leads both baselines by >= 6.0 without an error; the others all misaligned 10 to 40",
          [len(five), len(own), len(rest), all(parts(i)[3][0] == "m" for i in rest), rng([float(parts(i)[3][1:]) for i in rest], "1")],
          [5, 44, 11, True, "10 to 40"])
    return S


def mis(G, deg):
    th = np.deg2rad(deg)
    n = np.ones(3) / np.sqrt(3.0)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    Rm = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    if G == "SO3":
        return Rm
    if G == "SE2":
        M = np.eye(3)
        M[1:, 1:] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        return M
    M = np.zeros((6, 6))
    M[:3, :3] = Rm
    M[3:, 3:] = Rm
    return M


def first_cross(f, grid):
    v = f(grid)
    k = np.flatnonzero(v >= 1)
    if not len(k):
        return None
    a, b = grid[k[0] - 1], grid[k[0]]
    for _ in range(50):
        c = 0.5 * (a + b)
        a, b = (a, c) if f(np.array([c]))[0] >= 1 else (c, b)
    return b


def s_linearisation(S):
    """A = I + E(A0 - I) against the stored spr of every screened loop; stability margins in gain and misalignment."""
    A0 = {k: np.array(v["A"]) for k, v in J("review/F/derived.json")["A0"].items()}
    dmax, inside = 0.0, True
    for i, x in S.items():
        G, a, s, v = parts(i)
        A = A0[i.rsplit("__", 1)[0]]
        m = A.shape[0]
        g = np.ones(m)
        M = np.eye(m)
        if v[0] == "g":
            g = g * float(v[1:])
        elif v[0] == "a":
            g = np.random.default_rng(1000 + int(v[1:])).uniform(0.7, 1.3, m)
        else:
            M = mis(G, float(v[1:]))
        dmax = max(dmax, abs(spr(np.eye(m) + np.diag(g) @ M @ (A - np.eye(m))) - x["spr"]))
    kup, ang = {}, []
    for p, A in A0.items():
        m = A.shape[0]
        I, D = np.eye(m), A - np.eye(m)
        kup[p] = first_cross(lambda ks: spr(I[None] + ks[:, None, None] * D[None]), np.arange(1.0, 60.0, 1e-3))
        if not p.startswith("SO2"):
            for sg in (1, -1):
                ang.append(first_cross(lambda ts: spr(I[None] + np.stack([mis(p[:3], sg * t) for t in ts]) @ D[None]), np.arange(0.0, 180.0, 0.05)))
    gains = max(float(parts(i)[3][1:]) for i in S if parts(i)[3][0] == "g")
    bound("spr of I + E(A0 - I) against the stored spr of the 300 screened loops, largest difference", dmax, "lt", 3e-14, sci)
    check("constant-gain margin k_max over the 30 screened policies; largest screened gain inside",
          [len(kup), rng(list(kup.values()), "0.01"), gains < min(kup.values())], [30, "1.67 to 2.37", True])
    check("misalignment margins computed (24 policies outside SO(2), two senses)", len(ang), 48)
    bound("smallest misalignment margin in either sense, degrees", min(ang), "gt", 40, lambda v: q(v, "0.1"))
    sense = J("review/F/derived.json")["sense"]
    sh = lambda x: 100 * (x["ball_mu0"] - max(x["dist_mu0"], x["quad_mu0"]))
    se = [i for i in sense if parts(i)[0] in SE]
    plus_stored = {i for i in se if sh(S[i]) >= 5}
    plus = {i for i in se if sh(sense[i]["plus"]) >= 5}
    minus = {i for i in se if sh(sense[i]["minus"]) >= 5}
    check("opposite sense: misaligned SE(n) loops, larger baseline misses >= 5 points (screen sense, opposite), both, opposite only",
          [len(se), len(plus_stored), plus == plus_stored, len(minus), len(plus & minus), len(minus - plus)], [90, 32, True, 22, 18, 4])


def s_later():
    """Closed loops of the later screens: kept and refitted candidates."""
    P = {i: J(f"review/{'pert_r2' if os.path.exists(f'{R}/review/pert_r2/path/{i}.json') else 'pert_r3'}/path/{i}.json") for i in LATER}
    keep = [i for i in LATER if P[i]["carried_region_mu0"] >= P[i]["oracle_region_mu0"] - P[i]["tol"]]
    check("later screens: loops, kept (within 2 points of the oracle), refitted, refit larger on all refitted",
          [len(LATER), len(keep), sorted(keep) == sorted(CARRY), len(WARM), all(P[i]["warm_region_mu0"] >= P[i]["carried_region_mu0"] for i in WARM)],
          [29, 14, True, 15, True])
    k = J("review/F/kept14/kept_eig.json")["rows"]
    ids = sorted(f"{r['group']}__{r['dir']}__s{r['seed']}__{r['variant']}" for r in k)
    lam = [r["lam_max"] for r in k]
    check("kept candidates: the 14, eigenvalue of A'PA - P negative, range", [len(k), ids == sorted(CARRY), max(lam) < 0, rng(lam, "0.001")],
          [14, True, True, "-0.997 to -0.529"])
    ent = all(EV(i)["fns"]["V"]["entry_violations"] == 0 and S2(i)["fns"]["V"]["entry_violations"] == 0 for i in CARRY)
    check("kept candidates: sampled entry check holds on all 14", ent, True)


def s_without_error():
    """The region computed without the actuator error, checked under it."""
    N = {i: nominal(i) for i in I55}
    fails = {i: N[i]["decrease_violations"] + N[i]["containment_violations"] for i in I55}
    mi = [i for i in I55 if parts(i)[3][0] == "m"]
    by = lambda d: [sum(fails[i] > 0 for i in mi if parts(i)[3] == f"m{d}"), sum(parts(i)[3] == f"m{d}" for i in mi)]
    check("misaligned: loops failing a condition, of; at 5, 10, 15 degrees; at 30 and 40",
          [sum(fails[i] > 0 for i in mi), len(mi), by(5), by(10), by(15), [a + b for a, b in zip(by(30), by(40))]], [21, 32, [0, 4], [1, 6], [4, 6], [16, 16]])
    check("largest failing fraction of the points outside N_b", q(100 * max(N[i]["violation_fraction_of_region"] for i in I55)), "2.2")
    g = [i for i in I55 if parts(i)[3][0] == "g"]
    a = [i for i in I55 if parts(i)[3][0] == "a"]
    check("constant gain: failing loops of, failing points; per-axis gains: failing of",
          [sum(fails[i] > 0 for i in g), len(g), sum(fails[i] for i in g), sum(fails[i] > 0 for i in a), len(a)], [1, 14, 1, 5, 9])
    check("every sampled initial error in these regions converges", all(N[i]["nonconverged_in_nominal_region_mu0"] == 0 for i in I55), True)
    srch = []
    for i in LATER:
        p = f"review/pert_r2/eval_carry/{i}.json"
        c = J(p if os.path.exists(f"{R}/{p}") else f"review/pert_r3/eval_carry/{i}.json")["fns"]["V"]
        srch.append(c["lowered_by_search"] and c["level"] / (1 - 1e-9) <= N[i]["nominal_level"])
    check("later screens: the search finds failing points in this region, the samples", [sum(srch), sum(fails[i] > 0 for i in LATER), len(LATER)], [25, 11, 29])


def s_hinge():
    """The neural Lyapunov baseline: loops without a region and its fraction of the oracle region."""
    H = J("review/hinge_base/hb_levels.json")["levels"]
    zero = sorted(i for i in I55 if H[i]["region_mu0"] == 0)
    want = sorted([f"SE3__ppo_SE3_ls-1.5__s{s}__{v}" for s in range(3) for v in ("m30", "m40")] + ["SE3__ppo_SE3_ls-1.5__s1__g1.5", "SE3__trpo_SE3_ls-1.5__s1__m40"])
    lv = [H[i]["level"] for i in zero]
    ent = [H[i]["entry_violations"] for i in zero]
    check("V_h gives no region on: count, the listed loops, levels, entry failures",
          [len(zero), zero == want, f"{q(min(lv), '0.001')} to {q(max(lv))}", rng(ent, "1")], [8, True, "0.015 to 7.8", "12 to 72"])
    check("V_h median fraction of the oracle region", q(st.median(100 * 100 * H[i]["region_mu0"] / orc(i) for i in I55)), "73.0")


def s_checks():
    """Checks over the 94 closed loops: false inclusions, entry, failing fraction, rounds, lowerings."""
    runs = []
    for i in I94:
        for n in ("V", "Vd", "VP"):
            runs.append((i, n, EV(i)["fns"][n]))
            runs.append((i, n, S2(i)["fns"][n]))
        p = f"{R}/review/F/search2_from/{i}.json"
        if os.path.exists(p):
            runs.append((i, "V", J(p)["fns"]["V"]))
    fi = sum(r["false_inclusions_mu0"] + r["false_inclusions_ball"] for _, _, r in runs)
    ent = all(r["entry_violations"] == 0 and r["nb_nonconverged"] == 0 for _, n, r in runs if n == "V")
    check("no region of V, V_d, V_P holds a non-converging sampled initial error; V entry and N_b convergence on all 94", [fi, ent], [0, True])
    check("largest failing fraction of the ball outside N_b before the level (final candidates)", q(100 * max(EV(i)["fns"]["V"]["frac_decrease_violation_ball"] for i in I94), "0.01"), "0.90")
    r1 = max(len(EV(i)["fns"][n]["adversarial_rounds"]) for i in I94 for n in ("V", "Vd", "VP"))
    r2 = max(len(r["adversarial_rounds"]) for i, n, r in runs if r is not EV(i)["fns"][n])
    done = all(r["adversarial_converged"] and r["adversarial_rounds"][-1]["found"] == 0 for _, _, r in runs)
    check("rounds at most (first run, second run); every run ends with a round that found nothing", [r1, r2, done], [13, 6, True])
    l1 = [i for i in I94 if EV(i)["fns"]["V"]["lowered_by_search"]]
    l2 = [i for i in I94 if S2(i)["fns"]["V"]["lowered"]]
    big = [i for i in l2 if 100 * (FL[i]["V"]["region_first"] - S2(i)["fns"]["V"]["region_mu0"]) > 1]
    anyrun = set(l1) | set(l2) | {i for i in I94 if FL[i]["V"]["source"] == "search2_from"}
    check("first run lowered V below the sample level on, all SE(n); second run lowered further on, by > 1 point on; either run on",
          [len(l1), all(parts(i)[0] in SE for i in l1), len(l2), len(big), len(anyrun)], [59, True, 26, 2, 60])
    b, m = "SE3__ppo_SE3_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s0__m30"
    rb = {i: J(f"review/F/rbsens/{i}__rb0.2.json")["V"]["level"] for i in (b, m)}
    rc = J("review/F/rbcheck/SE3__ppo_SE3_ls-1.5__s1__base__rb0.2.json")
    pts = rc["inside"]
    gam = np.pi - np.pi / 8
    check("r_b = 0.2 run below the first run (s1 base, s0 m30), below the second run (s1 base yes, s0 m30 no); second run from it lowers s0 m30",
          [rb[b] < FL[b]["V"]["level_first"], rb[m] < FL[m]["V"]["level_first"], rb[b] < FL[b]["V"]["level_search2"], rb[m] < FL[m]["V"]["level_search2"],
           FL[m]["V"]["level"] < rb[m], FL[b]["V"]["level"] == rb[b]], [True, True, True, False, True, True])
    check("PPO s1 SE(3): two points at the edge of the ball failing containment, below both run levels",
          [rc["n_inside_paper_region"], {p["kind"] for p in pts} == {"containment"}, all(gam - np.linalg.norm(p["z"]) < 1e-3 for p in pts),
           max(p["V"] for p in pts) < min(FL[b]["V"]["level_first"], FL[b]["V"]["level_search2"])], [2, True, True, True])
    ns = {parts(i)[0]: EV(i)["fns"]["V"]["entry_samples"] for i in I94}
    check("entry samples of N_b (SO2, SE2, SO3, SE3)", [ns["SO2"], ns["SE2"], ns["SO3"], ns["SE3"]], [100000, 100000, 88577, 33064])
    z = J("l4dc/roa_SO2_seed0.json")
    check("samples of the first two stages", sci(z["n_samples_Z1"] + z["n_samples_Z2"]), "1.2e6")


def s_rloc():
    """Decrease near the goal, sampled on shells."""
    D = {}
    for f in glob.glob(f"{R}/review/F/rloc/*.json"):
        d = J(f)
        sh = d["shells"]
        r = 0.0
        for s in sh:
            if s["max_ratio"] > -d["c"]:
                break
            r = s["r"]
        D[os.path.basename(f)[:-5]] = dict(d, r_loc_from_shells=r, n_radii=len(sh))
    D.update(J("review/F/rloc_summary_SO2.json"))
    so = [k for k in D if k.startswith("SO")]
    sen = [k for k in D if k.startswith("SE")]
    same = all(abs(d["r_loc"] - d["r_loc_from_shells"]) == 0 for d in D.values())
    check("near goal: loops, c = 0.5 lambda_min(Q), r_loc recomputed from the shells, smallest r_loc, N_b inside on all",
          [len(D), sorted(D) == sorted(I94), all(abs(d["c"] - 0.5 * d["lambda_min_Q"]) < 1e-12 for d in D.values()), same, q(min(d["r_loc"] for d in D.values()), "0.001"),
           all(d["Nb_in_B_rloc"] for d in D.values())], [94, True, True, True, "0.105", True])
    check("near goal: shells and directions (outside SO(2)), radii on SO(2), shells span 1e-4 to gamma",
          [{(d["n_shells"], d["n_per_shell"]) for k, d in D.items() if not k.startswith("SO2")} == {(120, 20000)},
           sci(min(d["n_radii"] for k, d in D.items() if k.startswith("SO2"))), all(abs(d["r_last"] - d["gamma"]) < 1e-12 and abs(d["r_first"] - 1e-4) < 1e-12 for k, d in D.items() if k.startswith("SO2"))],
          [True, "1.1e6", True])
    check("N_b lies in the ball of radius (SO(n), SE(n))", [q(max(D[k]["r_Nb"] for k in so), "0.001"), q(max(D[k]["r_Nb"] for k in sen), "0.001")], ["0.071", "0.100"])
    check("contraction in the norm of P on N_b: loops; kept candidates among the 94",
          [sum(d["Nb_closed_by_contraction"] for d in D.values()), sum(not d["P_m_matches_certificate"] for d in D.values())], [89, 14])
    dv = J("review/F/derived.json")["Q"]
    check("computed Q on SO(3) and SE(3)", [[q(x, "0.001") for x in np.diag(dv["SO3"])], [q(x, "0.001") for x in np.diag(dv["SE3"])],
                                          bool(np.abs(np.array(dv["SE3"]) - np.diag(np.diag(dv["SE3"]))).max() < 1e-6)],
          [["2.000"] * 3, ["2.000"] * 3 + ["1.000"] * 3, True])
    check("volume-uniform point of the 6-dimensional ball within radius 0.1 of the goal", sci((0.1 / (np.pi - np.pi / 8)) ** 6, 0), "2e-9")


def s_critic():
    """The critic function v_c(Id) - v_c(Exp z) on the closed loops trained from scratch."""
    C = {i: EV(i)["fns"]["Vv"] for i in SCRATCH65 if "Vv" in EV(i)["fns"]}
    none = [i for i in C if C[i]["region_mu0"] == 0]
    p1, p2 = "SO2__ppo_SO2_ls-1.5__s1__base", "SO2__ppo_SO2_ls-1.5__s2__base"
    cap = [i for i in C if len(C[i]["adversarial_rounds"]) == 50]
    check("critic function: loops, no region on, samples in the PPO s1 SO(2) region, PPO s2 SO(2) percent, entry fails at all of",
          [len(C), len(none), round(C[p1]["region_mu0"] * 2e5), q(100 * C[p2]["region_mu0"]), C[p2]["entry_violations"] == C[p2]["entry_samples"], C[p2]["entry_samples"]],
          [50, 48, 1, "0.3", True, 13469])
    check("critic function: loops whose search used all 50 rounds, all on SO(2), last round still found points",
          [len(cap), all(i.startswith("SO2") for i in cap), all(C[i]["adversarial_rounds"][-1]["found"] > 0 for i in cap)], [5, True, True])


def s_correction():
    """Policy correction and the stability test over the 39 policies."""
    O = {i: extra(i)["offset"] for i in I39}
    mx = max(I39, key=lambda i: O[i]["max_increment_change_over_alpha_max"])
    med = [O[i]["median_increment_change"] / O[i]["alpha_max"] for i in I39]
    check("increment change at most (alpha_max), where; median below 0.14 on all, below 0.02 on; subtracted constant at most",
          [q(O[mx]["max_increment_change_over_alpha_max"], "0.01"), nice(mx), max(med) < 0.14, sum(x < 0.02 for x in med), q(max(O[i]["bias_norm"] for i in I39), "0.01")],
          ["0.17", "SO3 PPO s0 base", True, 24, "0.17"])
    sp = {i: extra(i)["spr"] for i in I39}
    w = max(I39, key=lambda i: sp[i]["max_abs_deviation_from_used"])
    check("spr changes with the step by at most, where; by less than 1e-6 on; steps",
          [sci(sp[w]["max_abs_deviation_from_used"]), nice(w), sum(sp[i]["max_abs_deviation_from_used"] < 1e-6 for i in I39),
           sorted(float(k) for k in sp[w]["values"])], ["1.4e-4", "SO2 SAC s0 base", 36, [1e-7, 1e-6, 1e-5, 1e-4]])
    check("largest spr among the 94 closed loops", q(max(EV(i)["spr"] for i in I94), "0.001"), "0.873")


def s_cost():
    """Training, evaluation, oracle and per-point times."""
    first = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{R}/final/eval/*.json"))
    tr = lambda G: [EV(i)["cert_wall_s"] / 3600 for i in first if i.startswith(G)]
    ev = lambda G: [EV(i)["eval_wall_s"] / 60 for i in first if i.startswith(G)]
    check("training and refinement hours on SO2, SO3, SE2, SE3 (first 47)", [len(first)] + [rng(tr(G)) for G in ("SO2", "SO3", "SE2", "SE3")],
          [47, "0.5 to 1.3", "0.8 to 1.3", "0.5 to 3.7", "0.8 to 5.0"])
    check("evaluation minutes on SO2, SO3, SE2, SE3", [rng(ev(G), "1") for G in ("SO2", "SO3", "SE2", "SE3")], ["7 to 22", "10 to 28", "19 to 73", "22 to 145"])
    p36 = [i for i in SCR55 if i not in first]
    check("training hours: the other 18 of the first screen; all 26 of the first screen",
          [len(p36), rng([EV(i)["cert_wall_s"] / 3600 for i in p36]), len(SCR55), rng([EV(i)["cert_wall_s"] / 3600 for i in SCR55])], [18, "4.0 to 6.7", 26, "2.9 to 6.7"])
    check("refit hours (15); hours per trained or refitted loop with an actuator error; minutes to keep a candidate (14)",
          [rng([EV(i)["cert_wall_s"] / 3600 for i in WARM]), rng([EV(i)["cert_wall_s"] / 3600 for i in WARM + SCR55]), rng([EV(i)["eval_wall_s"] / 60 for i in CARRY], "1")],
          ["2.5 to 4.3", "2.5 to 6.7", "41 to 156"])
    ow = [J(f"rx/oracle/{i}.json") for i in I94]
    check("oracle: minutes per closed loop; simulated steps", [rng([o["wall_s"] / 60 for o in ow], "1"), {o["N_rollout"] for o in ow}], ["18 to 48", {3000}])
    t = [extra(i)["timing"] for i in first]
    check("per point: V (microseconds), V_m with 600 and 3000 steps (milliseconds), threads",
          [rng([x["V_s_per_point"] * 1e6 for x in t]), rng([x["Vm600_s_per_point"] * 1e3 for x in t]), rng([x["Vm3000_s_per_point"] * 1e3 for x in t]), {str(x["threads"]) for x in t}],
          ["1.7 to 3.4", "0.5 to 1.0", "2.6 to 4.9", {"2"}])


def s_training():
    """kappa, refinement rounds, the gain-1.25 example, the validation of the later-screen procedure, non-finite losses."""
    k, k0, best = [], [], []
    for i in SCRATCH65:
        p = rel(EV(i)["cert"])
        c, c0 = PK(p), PK(p[:-4] + "__trained.pkl")
        k.append(kappa(c))
        k0.append(kappa(c0))
        best.append(c["refine_best_round"])
    check("kappa over the 65 trained from scratch, after and before the refinement", [len(k), rng(k, "0.01"), rng(k0, "0.01")], [65, "0.24 to 0.49", "0.24 to 0.41"])
    check("candidate before refinement kept on, refined on, after rounds", [best.count(0), sum(b > 0 for b in best), rng([b for b in best if b], "1")], [44, 21, "1 to 6"])
    i = "SE3__ppo_SE3_ls-1.5__s1__g1.25"
    c = PK(rel(EV(i)["cert"]))
    est = [100 * r["region"] for r in c["refine_log"]]
    check("PPO s1 gain 1.25 SE(3): quick estimates (count, range), full level rule for the kept and for the unrefined candidate",
          [len(est), rng(est), q(100 * FL[i]["V"]["region_mu0"]), q(100 * FX[i + "__norefine"]["region_mu0"])], [7, "53.5 to 64.8", "47.3", "47.5"])
    W = "review/warmcheck"
    five = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{R}/{W}/eval_carry/*.json"))
    diff = {}
    for j in five:
        cr = J(f"{W}/eval_carry/{j}.json")["fns"]["V"]["region_mu0"]
        reg = cr
        if cr < J(f"rx/oracle/{j}.json")["region_mu0"] - 0.02:
            reg = max(cr, J(f"{W}/eval_warm/{j}.json")["fns"]["V"]["region_mu0"])
        diff[j] = 100 * (reg - FL[j]["V"]["region_mu0"])
    s15 = "SE3__ppo_SE3_ls-1.5__s1__m15"
    m = re.search(r"best round (\d+) estimated region ([0-9.]+)", open(f"{R}/review/logs/warmcheck__{s15}.txt").read())
    check("validation on five first-screen loops: within 0.5 on, PPO s1 m15 SE(3) lower by, its quick estimate, its level-rule region",
          [len(five), sum(abs(d) <= 0.5 for d in diff.values()), q(-diff[s15]), q(100 * float(m.group(2))), q(100 * J(f"{W}/eval_warm/{s15}.json")["fns"]["V"]["region_mu0"])],
          [5, 4, "4.1", "64.3", "44.5"])
    a1 = "SE3__ppo_SE3_ls-1.5__s1__a1"
    blocks, cur, first_it = [], 0, None
    for l in open(f"{R}/review/logs/fast__{a1}.txt"):
        if l.startswith("{") and '"round"' in l:
            blocks.append(cur)
            cur = 0
        mm = re.search(r"it\s+(\d+)\s+non-finite loss.*\((\d+) so far\)", l)
        if mm:
            if first_it is None:
                first_it = (len(blocks), int(mm.group(1)))
            cur = int(mm.group(2))
    c = PK(rel(EV(a1)["cert"]))
    check("PPO s1 per-axis gains 1 SE(3): non-finite losses from refinement round (iteration), in every later round, kept round",
          [first_it[0], first_it[1] < 200, all(b > 0 for b in blocks[first_it[0]:]), c["refine_best_round"], len(blocks) - 1], [3, True, True, 6, 6])


def s_oracle_search():
    """Search of the oracle region for containment failures on three closed loops."""
    O = {n: J(f"review/oracle_search/{n}.json") for n in ("SE3_s0_m40", "SE3_s1_base", "SE2_s0_m40")}
    check("oracle search: fresh points, least W_m over the level (in this order)",
          [{sci(o["seed"]["n_fresh"], 0) for o in O.values()}, [q(o["S1"]["ratio_to_level"], "0.001") for o in O.values()]], [{"2e7"}, ["0.885", "0.559", "0.932"]])
    a = O["SE3_s0_m40"]
    i = a["id"]
    v = 100 * FL[i]["V"]["region_mu0"]
    check("PPO s0 m40 SE(3): oracle region before and after, candidate fraction after and before",
          [q(100 * a["oracle_region_mu0"], "0.01"), q(100 * a["mu0"]["region_mu0_at_wstar"], "0.01"), q(100 * v / (100 * a["mu0"]["region_mu0_at_wstar"])),
           q(100 * v / (100 * a["oracle_region_mu0"]))], ["63.47", "60.53", "98.2", "93.7"])
    check("oracle region falls on the third and second closed loops by", [q(100 * O["SE2_s0_m40"]["mu0"]["affected_mu0"], "0.01"), O["SE3_s1_base"]["mu0"]["affected_mu0"]],
          ["0.06", 0.0])
    th = [o["mu0_thresholds"] for o in O.values() if o["mu0"]["affected_mu0"] > 0]
    check("most of the reduction from the 2e7 points, not the random steps; 100 times the level samples",
          [all(t["oracle_level"]["region_mu0"] - t["fresh_samples_only"]["region_mu0"] > t["fresh_samples_only"]["region_mu0"] - t["after_local_search"]["region_mu0"] for t in th),
           round(O["SE3_s0_m40"]["seed"]["n_fresh"] / 2e5)], [True, 100])
    check("oracle search minutes per closed loop on one thread", rng([o["wall_s"] / 60 for o in O.values()], "1"), "21 to 32")
    G = [json.loads(l) for l in open(f"{R}/review/oracle_search/grad.jsonl")]
    store = [g for g in G if g["mode"] == "wm" and g["n"] == 3000 and not g["remat"] and g["batch"] == 256][0]
    safe = [g for g in G if g["mode"] == "wmsafe" and g["n"] == 3000 and g["batch"] == 1024][0]
    cand = [g for g in G if g["mode"] == "cand" and g["batch"] == 32768 and g["name"] == "SE3_s0_m40"][0]
    about("stored trajectory, megabytes per point", store["xla_temp_MB"] / store["batch"], 9)
    about("gradient of W_m, milliseconds per point", safe["per_point_ms"], 20)
    check("gradient of delta V, milliseconds per point", q(cand["per_point_ms"], "0.001"), "0.024")
    about("one round of the third stage on the candidate, minutes", 32768 * 1000 * 2 * cand["per_point_ms"] / 1e3 / 60, 26)
    about("one round on the oracle, days", 32768 * 1000 * safe["per_point_ms"] / 1e3 / 86400, 8)


def s_ablations():
    """Text of the ablation paragraphs not printed by ablation2.py."""
    so = ["SO2__trpo_SO2_ls-1.5__s2__base", "SO3__td3_SO3_ls-1.5__s1__base"]
    check("z'P_m z on SO(n) has the region of V_P", [FX[i + "__VPm"]["region_mu0"] == FL[i]["VP"]["region_mu0"] for i in so], [True, True])
    g = "SE3__ppo_SE3_ls-1.5__s1__g1.5"
    h = FX[g + "__hinge"]
    he = J(f"review/ablation/eval/{g}__hinge.json")["fns"]["V"]
    check("hinge-only variant with gain 1.5: region, fails the entry check", [h["region_mu0"], he["entry_violations"] > 0], [0.0, True])
    five = ["SO3__td3_SO3_ls-1.5__s1__base", "SE2__ppo_SE2_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s2__base", g]
    kb = {(i, v): PK(f"review/ablation/cert/{i}__{v}.pkl")["refine_best_round"] for i in five for v in ("hinge", "regress")}
    check("hinge-only and regression keep the candidate before refinement, except regression on PPO s1 SE(3) without error and with gain 1.5",
          sorted(nice(i) + " " + v for (i, v), b in kb.items() if b > 0), ["SE3 PPO s1 base regress", "SE3 PPO s1 g1.5 regress"])
    s2 = "SE3__ppo_SE3_ls-1.5__s2__base"
    blocks, cur = [], 0
    for l in open(f"{R}/review/logs/abl_regress__{s2}.txt"):
        if l.startswith("{") and '"round"' in l:
            blocks.append(cur)
            cur = 0
        if "non-finite" in l:
            cur += 1
    c = PK(f"review/ablation/cert/{s2}__regress.pkl")
    reg = [r["region"] for r in c["refine_log"]]
    first = next(k for k, b in enumerate(blocks) if b > 0)
    check("regression on PPO s2 SE(3): non-finite from refinement round, kept round, rounds in between smaller",
          [first, all(b > 0 for b in blocks[first:]), c["refine_best_round"], all(r < reg[0] for r in reg[1:first])], [4, True, 0, True])


def s_sensitivity():
    """Sensitivity to r_b and r_a."""
    ids = sorted({os.path.basename(p).rsplit("__rb", 1)[0] for p in glob.glob(f"{R}/review/F/rbsens/*.json")})
    reg, ok = {}, True
    for i in ids:
        a, b = J(f"review/F/rbsens/{i}__rb0.05.json")["V"], J(f"review/F/rbsens/{i}__rb0.2.json")["V"]
        ok &= a["checks_pass"] and b["checks_pass"]
        reg[i] = [100 * a["region_mu0"], 100 * FL[i]["V"]["region_first"], 100 * b["region_mu0"]]
    sp = {i: max(v) - min(v) for i, v in reg.items()}
    flat = ["SO3__td3_SO3_ls-1.5__s1__base", "SE2__ppo_SE2_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s2__base"]
    two = ["SE3__ppo_SE3_ls-1.5__s0__m30", "SE3__ppo_SE3_ls-1.5__s1__g1.5"]
    check("r_b: loops, all checks pass, within 0.003 on three, at most on two, PPO s1 SE(3) at 0.05, 0.1, 0.2",
          [len(ids), ok, max(sp[i] for i in flat) <= 0.003, max(sp[i] for i in two) <= 0.23, [q(x) for x in reg["SE3__ppo_SE3_ls-1.5__s1__base"]]],
          [6, True, True, True, ["62.0", "60.5", "52.5"]])
    ra = ["SE3__ppo_SE3_ls-1.5__s0__m30", "SE3__ppo_SE3_ls-1.5__s1__base", "SE3__ppo_SE3_ls-1.5__s1__g1.25"]
    check("r_a = 0.15 against 0.3 (one search run)", [[q(100 * J(f"review/F/ra015/eval/{i}__ra015.json")["fns"]["V"]["region_mu0"]) for i in ra],
                                                       [q(100 * FL[i]["V"]["region_first"]) for i in ra]], [["64.2", "59.9", "58.2"], ["64.6", "60.5", "47.3"]])


def s_seeds():
    """A second candidate trained with another seed on the 16 closed loops."""
    d = {}
    for p in glob.glob(f"{R}/review/seeds/eval/*.json"):
        i = os.path.basename(p).rsplit("__cs", 1)[0]
        if i in FL:
            d[i] = 100 * (J(p)["fns"]["V"]["region_mu0"] - FL[i]["V"]["region_first"])
    big = sorted((i for i in d if abs(d[i]) > 1), key=lambda i: ["base", "g1.25", "g1.5"].index(parts(i)[3]))
    check("seeds: loops, median and mean difference, loops differing by > 1, their differences, largest on the others",
          [len(d), q(st.median(d.values()), "0.01"), q(st.mean(d.values()), "0.01"), [nice(i) for i in big], [("+" if d[i] > 0 else "") + q(d[i], "0.01") for i in big],
           q(max(abs(d[i]) for i in d if i not in big), "0.01")],
          [16, "0.00", "0.22", ["SE3 PPO s1 base", "SE3 PPO s1 g1.25", "SE3 PPO s1 g1.5"], ["+3.53", "+1.97", "-1.15"], "0.69"])
    gap = []
    for v in ("g1.25", "g1.5"):
        i = f"SE3__ppo_SE3_ls-1.5__s1__{v}"
        c = PK(rel(EV(i)["cert"]))
        gap.append(c["refine_log"][c["refine_best_round"]]["region"] > FL[i]["V"]["region_first"])
        m = re.search(r"best round \d+ estimated region ([0-9.]+)", open(f"{R}/review/logs/seed1__{i}.txt").read())
        gap.append(float(m.group(1)) > J(glob.glob(f"{R}/review/seeds/eval/{i}__cs*.json")[0])["fns"]["V"]["region_mu0"])
    check("quick estimate exceeds the level-rule region with either seed (gain 1.25, 1.5)", gap, [True] * 4)


def s_deploy():
    """Deployment on the CubeSat and the BlueROV2."""
    C, B = J("review/F/deploy/cubesat.json"), J("review/F/deploy/bluerov2.json")
    arms = [x for d in (C, B) for x in d["arms"].values()]
    check("levels of the two deployed candidates", [q(C["rho"], "0.01"), q(B["rho"], "0.01"), C["rho"] == FL["SO3__ppo_SO3_ls-1.5__s0__base"]["V"]["level"],
                                                     B["rho"] == FL["SE3__ppo_SE3_ls-1.5__s0__base"]["V"]["level"]], ["44.40", "44.08", True, True])
    check("deployment: episodes, policy steps, exits in every setting (nominal, two initial rates, halved period)",
          [{a["episodes"] for a in arms}, {C["steps"], B["steps"]}, sum(a["exits"] for a in arms), len(arms)], [{200}, {40}, 0, 8])
    check("halved period: carried-out percent on the CubeSat and the BlueROV2; largest increase of V in the extra settings",
          [q(100 * C["arms"]["half"]["carried_out_median"], "1"), q(100 * B["arms"]["half"]["carried_out_median"], "1"),
           sci(max(a["largest_V_increase"] for d in (C, B) for k, a in d["arms"].items() if k != "nominal"), 0)], ["73", "96", "2e-6"])
    n = C["arms"]["nominal"], B["arms"]["nominal"]
    check("deployment table: largest increase of V, carried out, final error (CubeSat, BlueROV2)",
          [sci(n[0]["largest_V_increase"]), q(n[0]["carried_out_median"], "0.001"), sci(n[0]["final_angle_median"]),
           sci(n[1]["largest_V_increase"]), q(n[1]["carried_out_median"], "0.001"), sci(n[1]["final_error_median"])],
          ["1.4e-22", "0.936", "7.6e-15", "8.2e-32", "1.000", "1.8e-16"])
    check("BlueROV2: saturated thruster percent, period, gains", [q(100 * n[1]["saturated_substep_fraction"]), B["dt_outer"], B["kp_pos"], B["kd_pos"]], ["0.6", 2.0, 400.0, 160.0])
    m, dims = 4.0, (0.10, 0.10, 0.34)
    box = [m * (dims[1] ** 2 + dims[2] ** 2) / 12, m * (dims[0] ** 2 + dims[1] ** 2) / 12]
    t = 2 * math.sqrt(J("l4dc/roa_SO3_seed0.json")["alpha_max"] * 0.049 / 1e-3)
    check("CubeSat: inertia above that of a uniform box (percent), rest-to-rest time, sampling period",
          [f"{q(100 * (0.049 / box[0] - 1), '1')} to {q(100 * (0.008 / box[1] - 1), '1')}", q(t, "0.01"), C["dt_outer"]], ["17 to 20", "7.85", 15.7])
    ppo = [i for i in I55 if parts(i)[0] == "SE3" and parts(i)[1] == "ppo"]
    rec = {i: J(f"review/rov/{i}__recomputed.json") for i in ppo}
    nomr = {i: J(f"review/rov_T/{i}__nominal.json") if os.path.exists(f"{R}/review/rov_T/{i}__nominal.json") else J(f"review/rov/{i}__nominal.json") for i in ppo}
    low = [i for i in ppo if FL[i]["V"]["source"] != "eval"]
    check("BlueROV2 with an actuator error: loops, exits and rises above 1e-12 (closed loop and vehicle), loops lowered by the second run started at the first-run level",
          [len(ppo), sum(r[k]["exits"] + r[k]["rises_above_floor"] for r in rec.values() for k in ("control", "plant")), len(low),
           all(rec[i]["rho"] == FL[i]["V"]["level_first"] for i in low)], [28, 0, 15, True])
    check("BlueROV2 with an actuator error: carried-out medians, saturated percent",
          [rng([r["plant"]["executed_median"] for r in rec.values()], "0.0001"), rng([100 * r["plant"]["saturated_substep_fraction"] for r in rec.values()], "0.01")],
          ["0.9999 to 1.0018", "0.27 to 0.71"])
    N = {i: nominal(i) for i in ppo}
    fail = [i for i in ppo if N[i]["decrease_violations"] + N[i]["containment_violations"] > 0]
    check("region without the error on these 28: fails on, at up to percent of its points outside N_b",
          [len(fail), q(100 * max(N[i]["violation_fraction_of_region"] for i in ppo), "0.01")], [10, "0.26"])
    a, b = "SE3__ppo_SE3_ls-1.5__s0__m40", "SE3__ppo_SE3_ls-1.5__s1__m40"
    w = lambda r: max(r["control"]["worst_rise"], r["plant"]["worst_rise"])
    rest = [i for i in ppo if i not in (a, b)]
    check("region without the error: PPO s0 m40 exits and rises (closed loop, vehicle), worst rise; PPO s1 m40 exits, rises, worst rise; the other 26",
          [nomr[a]["control"]["exits"], nomr[a]["plant"]["exits"], nomr[a]["control"]["rises_above_floor"], q(w(nomr[a])),
           nomr[b]["control"]["exits"] + nomr[b]["plant"]["exits"], nomr[b]["control"]["rises_above_floor"], q(w(nomr[b]), "0.01"),
           len(rest), sum(nomr[i][k]["exits"] + nomr[i][k]["rises_above_floor"] for i in rest for k in ("control", "plant"))],
          [1, 1, 5, "3.8", 0, 1, "0.62", 26, 0])


def s_settings():
    """Constants of the training table and of the level rule, read from the stored configurations."""
    cf = [PK(rel(EV(i)["cert"])) for i in SCRATCH65]
    keys = dict(hidden=(256, 256), feat_out=16, alpha_floor=0.05, alpha_init=0.3, eps_ball=0.1, zeta_train=0.01, lr=1e-3, batch=4096, buffer_size=32768,
                resample_every=200, shell_batch=2048, w_shell=0.3, w_spec=1e-3, spec_target=2.0, lam_lr=0.02, lam_max=50.0, pgd_every=10, mine_batch=2 ** 18,
                mine_keep=2048, pgd_batch=1024, pgd_steps=30, ce_buffer_max=16384, ce_batch=1024, w_ce=5.0, w_exit=1.0, exit_margin=1.2, radial_frac=0.5)
    bad = sorted({k for c in cf for k, v in keys.items() if c["cfg"][k] != v})
    it = {("SO" if c["group"].startswith("SO") else "SE"): set() for c in cf}
    for c in cf:
        it["SO" if c["group"].startswith("SO") else "SE"].add(c["cfg"]["iters"])
    rs = {tuple(sorted(c["refine_settings"].items())) for c in cf}
    check("training constants of Table consts on the 65 trained from scratch: differing keys, iterations (SO, SE), refinement settings, gamma, r_a",
          [bad, sorted(it["SO"]), sorted(it["SE"]), rs == {(("iters_per", 1000), ("jitter", 0.05), ("n_copies", 8), ("rounds", 6))},
           {q(c["cfg"]["radius"], "0.000001") for c in cf} == {q(np.pi - np.pi / 8, "0.000001")}, {c["anchor"]["r_a"] for c in cf}], [[], [8000], [4000], True, True, {0.3}])
    s2 = [S2(i) for i in I94]
    check("level rule: search starts, Adam steps, round cap, simulated steps, r_b",
          [{x["adv_n"] for x in s2}, {x["adv_steps"] for x in s2}, {x["adv_rounds_cap"] for x in s2}, {x["n_roll"] for x in s2}, {x["r_b"] for x in s2}],
          [{32768}, {1000}, {50}, {600}, {0.1}])
    b = {d: {x["budget"] for x in J(f"{d}/summary.json")} for d in (f"{a}_{G}_ls-1.5" for a in ("ppo", "trpo", "sac", "td3") for G in ("SO2", "SO3", "SE2", "SE3"))}
    check("environment steps: PPO and TRPO, SAC and TD3", [set().union(*(v for k, v in b.items() if k[:3] in ("ppo", "trp"))), set().union(*(v for k, v in b.items() if k[:3] in ("sac", "td3")))],
          [{400000}, {128000}])
    nf = {i: extra(i)["fit"]["600"]["n"] for i in I94}
    p = math.acos(1 - 0.1 ** 2 / 4) / (np.pi - np.pi / 8)
    so2 = {n for i, n in nf.items() if i.startswith("SO2")}
    check("fit sample: 2e4 points drawn in the ball, those outside N_b kept (all other groups 20000; SO(2) within 3 sd of 2e4 (1 - |N_b|))",
          [{n for i, n in nf.items() if not i.startswith("SO2")}, all(abs(n - 2e4 * (1 - p)) < 3 * math.sqrt(2e4 * p * (1 - p)) for n in so2)], [{20000}, True])
    d = J("review/F/so2cert_v3/SO2__ppo_SO2_ls-1.5__s0__base.json")
    check("N_b on SO(2) is |z| < arccos(1 - r_b^2/4)", q(d["local_delta"], "0.0000001"), q(math.acos(1 - 0.1 ** 2 / 4), "0.0000001"))
    check("clip length alpha_max = 0.1 phi_bar on the rigid-motion groups", q(0.1 * math.sqrt(np.pi ** 2 + (np.pi / 2) ** 2), "0.001"), "0.351")


def main():
    s_policies()
    s_ball()
    s_levels_fit()
    s_heldout()
    s_so2proof()
    s_fixed_point()
    s_nominal_td3()
    S = s_screens()
    s_linearisation(S)
    s_later()
    s_without_error()
    s_hinge()
    s_checks()
    s_rloc()
    s_critic()
    s_correction()
    s_cost()
    s_training()
    s_oracle_search()
    s_ablations()
    s_sensitivity()
    s_seeds()
    s_deploy()
    s_settings()
    for ok, name, got, want in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: computed {got}, paper {want}")
    nf = sum(not r[0] for r in results)
    print(f"{len(results) - nf} PASS, {nf} FAIL")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
