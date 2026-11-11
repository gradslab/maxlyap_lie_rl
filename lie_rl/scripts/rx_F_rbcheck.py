"""Re-run the first search round of rx_F_rbsens.py with r_b = 0.2 on one loop, exactly (same samples,
same random generator, same starts), keep every point the round flags, and recheck each one directly in float64:
V(z), ell(Exp z), the one-step change V(F(z)) - V(z) and |F(z)|. A flagged point with V(z) below the level of the
paper's region (r_b = 0.1) and ell >= 0.1 that fails decrease or containment lies in that region and fails a condition.

Usage: JAX_ENABLE_X64=1 python scripts/rx_F_rbcheck.py GROUP DIR SEED VARIANT EVAL_JSON R_B OUT_JSON
"""
import json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import jax, jax.numpy as jp, numpy as np, optax
from lie_rl import anchor as AN, certificate as C, variants as Vr
import rx_F_common as F

group, d, seed, variant, eval_path, r_b, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], float(sys.argv[6]), sys.argv[7]
t0 = time.time(); ev = json.load(open(eval_path)); rho_paper = ev["fns"]["V"]["level"]
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(F.cert_of(eval_path), "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius
AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G); fn = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
geo, _ = F.geometry(m, gamma, Tj, ell)
H = []
for g in geo:
    fz, fF = F.evalb(g["Z"], fn), F.evalb(g["FZ"], fn)
    H.append(dict(f=fz, bad=((g["l"] >= r_b) & (fF - fz >= 0)) | (g["rF"] > gamma)))
rho = float(H[0]["f"].max())
for h in H:
    if h["bad"].any(): rho = min(rho, (1 - F.ETA) * float(h["f"][h["bad"]].min()))
rng = np.random.default_rng(seed + 17)
pool = np.concatenate([g["Z"][(h["f"] <= rho) & (g["l"] >= r_b)] for g, h in zip(geo, H)])
starts = pool[rng.choice(len(pool), min(32768, len(pool)), replace=False)]; scale = max(abs(rho), 1e-12)
def feas(zq, penalty):
    if penalty:
        return jax.nn.relu(fn(zq) - rho) / scale + jax.nn.relu(r_b - ell(zq)) + jax.nn.relu(jp.linalg.norm(zq, axis=-1) - gamma)
    return (fn(zq) <= rho) & (ell(zq) >= r_b) & (jp.linalg.norm(zq, axis=-1) <= gamma)
hits = []
for kind, score in (("decrease", lambda zq: fn(T(zq)) - fn(zq)), ("containment", lambda zq: jp.linalg.norm(T(zq), axis=-1) - gamma)):
    obj = lambda zq, score=score: -jp.sum(score(zq) - 100.0 * feas(zq, True))
    opt = optax.adam(3e-3); z = jp.asarray(starts); st = opt.init(z)
    @jax.jit
    def upd(zq, st, obj=obj):
        g_ = jax.grad(obj)(zq); u, st = opt.update(g_, st); return optax.apply_updates(zq, u), st
    for i in range(1000):
        z, st = upd(z, st)
        if (i + 1) % 25 == 0:
            f = np.asarray(feas(z, False)); s = np.asarray(score(z)); hit = f & (s >= 0)
            for zz in np.asarray(z)[hit]: hits.append((kind, zz))
out = []
for kind, zz in hits:
    zq = jp.asarray(zz[None]); Fz = Tj(zq)
    v, vF, l, nF = float(fn(zq)[0]), float(fn(Fz)[0]), float(ell(zq)[0]), float(jp.linalg.norm(Fz, axis=-1)[0])
    out.append(dict(kind=kind, z=[float(x) for x in zz], V=v, dV=vF - v, ell=l, normF=nF, norm=float(np.linalg.norm(zz))))
inside = [o for o in out if o["V"] <= rho_paper and o["ell"] >= 0.1 and o["norm"] <= gamma and (o["dV"] >= 0 or o["normF"] > gamma)]
rec = dict(group=group, dir=d, seed=seed, variant=variant, r_b=r_b, rho_round1=rho, rho_paper=rho_paper, gamma=gamma,
           n_flagged=len(out), n_inside_paper_region=len(inside),
           least_V_flagged=min((o["V"] for o in out), default=None),
           inside=sorted(inside, key=lambda o: o["V"])[:200], wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out_path, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("n_flagged", "n_inside_paper_region", "least_V_flagged", "rho_paper", "wall_s")}))
if inside: print("least-V violation inside the paper region:", json.dumps(sorted(inside, key=lambda o: o["V"])[0]))
