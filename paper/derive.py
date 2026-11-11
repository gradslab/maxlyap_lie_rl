"""Quantities that need the trained policies, written to results/review/F/derived.json for numbers.py:
the Jacobian A0 = DF(0) of every screened policy without the actuator error, the matrix Q of each group, the first two
stages of the baseline levels of the 90 misaligned screened closed loops in both senses of rotation (as rx_screen.py,
no rollouts), and the second fixed point of TRPO seed 2 on SO(2).
Run with the environment of lie_rl/requirements_results.txt:  JAX_ENABLE_X64=1 python derive.py
"""
import glob
import json
import os
import sys

from paths import REPO, RES

sys.path.insert(0, str(REPO / "lie_rl"))
os.chdir(REPO / "lie_rl")
import jax
import jax.numpy as jp
import numpy as np
from lie_rl import anchor as AN, certificate as C, roa as R, variants as Vr

assert jax.config.jax_enable_x64
GAMMA, RB, ETA, NSIM, NZ = float(np.pi - np.pi / 8), 0.1, 1e-9, 100000, 200000
out = dict(A0={}, Q={}, sense={}, fixed_point={})
scr = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "rx" / "screen" / "*.json")))]
pols = sorted({(r["group"], r["dir"], r["seed"]) for r in scr})
for p in pols:
    cl = Vr.ClosedLoop(*p, "base")
    A, dA = R.jacobian_with_uncertainty(cl.chart_map(), cl.m)
    out["A0"]["__".join([p[0], p[1], f"s{p[2]}"])] = dict(A=A.tolist(), unc=dA)
for G, d in (("SO2", "ppo_SO2_ls-1.5"), ("SO3", "ppo_SO3_ls-1.5"), ("SE2", "ppo_SE2_ls-1.5"), ("SE3", "ppo_SE3_ls-1.5")):
    cl = Vr.ClosedLoop(G, d, 0, "base")
    out["Q"][G] = np.asarray(AN.local_quadratic(cl.chart_map(), cl.G, cl.m)[2]).tolist()


def evalb(z, fn, bs=50000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])


cache = {}


def samples(cl):
    if cl.group not in cache:
        m = cl.m
        k = jax.random.split(jax.random.key(11), 6)
        X0 = cl.G.sample(k[0], NSIM, 1.0)
        z0 = np.asarray(cl.G.Log(X0))
        r0 = np.linalg.norm(z0, axis=-1)
        zb = np.asarray(C.sample_ball(k[1], NSIM, m, GAMMA))
        Z = np.concatenate([np.asarray(C.sample_ball(k[2], NZ, m, GAMMA)), np.asarray(C.sample_ball_radial(k[3], NZ, m, GAMMA))])
        cache[cl.group] = dict(z0=z0, inball=r0 <= GAMMA, r0=r0, zb=zb, Z=Z, lz=evalb(Z, C.make_ell(cl.G)), r=np.linalg.norm(Z, axis=-1))
    return cache[cl.group]


def screen(cl):
    S = samples(cl)
    T = jax.jit(cl.chart_map())
    A, _ = R.jacobian_with_uncertainty(cl.chart_map(), cl.m)
    spr = float(np.max(np.abs(np.linalg.eigvals(A))))
    Z, lz, r = S["Z"], S["lz"], S["r"]
    FZ = evalb(Z, T)
    rF = np.linalg.norm(FZ, axis=-1)
    bad_d = (lz >= RB) & (rF >= r)
    r_d = min((1 - ETA) * float(r[bad_d].min()) if bad_d.any() else GAMMA, GAMMA)
    rec = dict(spr=spr, ball_mu0=float(S["inball"].mean()), dist_mu0=float((S["r0"] <= r_d).mean()))
    P = R.stein(A)
    VP = lambda zz: np.einsum("ni,ij,nj->n", zz, P, zz)
    VZ, VF = VP(Z), VP(FZ)
    bad_q = ((lz >= RB) & (VF >= VZ)) | (rF > GAMMA)
    c_q = float(VZ.max())
    if bad_q.any():
        c_q = min(c_q, (1 - ETA) * float(VZ[bad_q].min()))
    rec["quad_mu0"] = float(((VP(S["z0"]) <= c_q) & S["inball"]).mean())
    return rec


for r in scr:
    v = r["variant"]
    if not v.startswith("m") or float(v[1:]) > 40:
        continue
    o = {}
    for sense in (1, -1):
        cl = Vr.ClosedLoop(r["group"], r["dir"], r["seed"], v)
        cl.M = jp.asarray(Vr.mis_matrix(r["group"], sense * float(v[1:])))
        o["plus" if sense > 0 else "minus"] = screen(cl)
    out["sense"][r["id"]] = o
    print(r["id"], {k: round(100 * (x["ball_mu0"] - max(x["dist_mu0"], x["quad_mu0"])), 3) for k, x in o.items()}, flush=True)

cl = Vr.ClosedLoop("SO2", "trpo_SO2_ls-1.5", 2, "base")
f = lambda z: float(np.asarray(cl.chart_map()(jp.asarray([[z]])))[0, 0] - z)
zs = np.linspace(2.69, 2.70, 1001)
fs = np.array([f(z) for z in zs])
i = int(np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:]))[0])
a, b = zs[i], zs[i + 1]
for _ in range(80):
    c = 0.5 * (a + b)
    a, b = (c, b) if np.sign(f(c)) == np.sign(f(a)) else (a, c)
out["fixed_point"] = dict(id="SO2__trpo_SO2_ls-1.5__s2__base", z=0.5 * (a + b), f_left=f(zs[i] - 1e-3), f_right=f(zs[i + 1] + 1e-3))
json.dump(out, open(RES / "review" / "F" / "derived.json", "w"), indent=1)
print("fixed point", out["fixed_point"])
