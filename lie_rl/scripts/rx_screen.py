"""Cheap screen of a closed loop: is there room for a learned Lyapunov function to beat the baselines?

Usage: JAX_ENABLE_X64=1 python scripts/rx_screen.py JOBFILE [LANE NLANES]
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, roa as R, variants as Vr

OUT = Path(__file__).parents[1] / "results" / "rx" / "screen"; OUT.mkdir(parents=True, exist_ok=True)
jobs = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
lane, nlanes = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
GAMMA, RB, ETA, NSIM, NZ = float(np.pi - np.pi / 8), 0.1, 1e-9, 100000, 200000


def job_id(G, d, s, v):
    return f"{G}__{d.replace('/', '-')}__s{s}__{v}"


def evalb(z, fn, bs=50000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])


for j, (G, d, s, v) in enumerate(jobs):
    if j % nlanes != lane: continue
    s = int(s); jid = job_id(G, d, s, v); path = OUT / f"{jid}.json"
    if path.exists(): continue
    t0 = time.time()
    cl = Vr.ClosedLoop(G, d, s, v); m = cl.m; T = cl.chart_map(); Tj = jax.jit(T); stepj = jax.jit(cl.step)
    A, dA = R.jacobian_with_uncertainty(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A))))
    sv = float(np.linalg.norm(A, 2))

    def roll(X, n=600):
        for _ in range(n // 50):
            X = jax.lax.fori_loop(0, 50, lambda i, Y: cl.step(Y), X)
        return X
    roll = jax.jit(roll)

    def conv(X):
        th, tr = cl.G.err_parts(roll(X)); return np.asarray((th < 0.05) & (tr < 0.05))
    k = jax.random.split(jax.random.key(11), 6)
    X0 = cl.G.sample(k[0], NSIM, 1.0); c0 = np.concatenate([conv(X0[i:i + 25000]) for i in range(0, NSIM, 25000)])
    r0 = np.linalg.norm(np.asarray(cl.G.Log(X0)), axis=-1); inball = r0 <= GAMMA
    zb = np.asarray(C.sample_ball(k[1], NSIM, m, GAMMA))
    cb = np.concatenate([conv(cl.G.Exp(jp.asarray(zb[i:i + 25000]))) for i in range(0, NSIM, 25000)])
    Z = np.concatenate([np.asarray(C.sample_ball(k[2], NZ, m, GAMMA)), np.asarray(C.sample_ball_radial(k[3], NZ, m, GAMMA))])
    ell = C.make_ell(cl.G); lz = evalb(Z, ell); FZ = evalb(Z, Tj); r = np.linalg.norm(Z, axis=-1); rF = np.linalg.norm(FZ, axis=-1)
    bad_d = (lz >= RB) & (rF >= r)
    r_d = (1 - ETA) * float(r[bad_d].min()) if bad_d.any() else GAMMA
    r_d = min(r_d, GAMMA)
    rec = dict(group=G, dir=d, seed=s, variant=v, id=jid, spr=spr, spr_uncertainty=dA, sigma_max=sv,
               passes_stability=bool(spr <= 0.99),
               true_mu0=float(c0.mean()), ball_mu0=float(inball.mean()), true_in_ball_mu0=float((c0 & inball).mean()),
               true_ball_fraction=float(cb.mean()),
               frac_decrease_viol_dist=float(bad_d[lz >= RB].mean()),
               dist_radius=r_d, dist_mu0=float((r0 <= r_d).mean()), dist_ball_fraction=float((r_d / GAMMA) ** m))
    if spr < 1.0:
        P = R.stein(A); VP = lambda zz: np.einsum("ni,ij,nj->n", zz, P, zz)
        VZ, VF = VP(Z), VP(FZ); bad_q = ((lz >= RB) & (VF >= VZ)) | (rF > GAMMA)
        c_q = float(VZ.max())
        if bad_q.any(): c_q = min(c_q, (1 - ETA) * float(VZ[bad_q].min()))
        z0 = np.asarray(cl.G.Log(X0))
        rec.update(quad_level=c_q, quad_mu0=float(((VP(z0) <= c_q) & inball).mean()),
                   quad_ball_fraction=float((VP(zb) <= c_q).mean()))
    else:
        rec.update(quad_level=None, quad_mu0=0.0, quad_ball_fraction=0.0)
    rec["headroom_mu0"] = rec["true_in_ball_mu0"] - max(rec["dist_mu0"], rec["quad_mu0"])
    rec["headroom_ball"] = rec["true_ball_fraction"] - max(rec["dist_ball_fraction"], rec["quad_ball_fraction"])
    rec["wall_s"] = time.time() - t0
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in rec.items()
                      if k_ in ("id", "spr", "true_in_ball_mu0", "ball_mu0", "dist_mu0", "quad_mu0", "headroom_mu0", "wall_s")}), flush=True)
