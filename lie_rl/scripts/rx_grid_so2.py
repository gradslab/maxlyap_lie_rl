"""Dense grid check on SO(2), where the coordinate ball is the interval [-gamma, gamma]: decrease and containment of the
final candidate at its final level on 2e7 + 1 equally spaced points (spacing about 2.7e-7). If points fail, the level
is lowered to (1 - 1e-9) times the least V at them, as in the level rule, and the region is recomputed on the mu_0
sample of rx_eval.py. Usage: JAX_ENABLE_X64=1 python scripts/rx_grid_so2.py ID OUT_JSON
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, variants as Vr, anchor as AN
i, out = sys.argv[1], sys.argv[2]; G_, d, s, v = i.split("__"); assert G_ == "SO2"
F_ = Path(__file__).parents[1] / "results" / "final"
cert = pickle.load(open(F_ / "cert" / f"{i}.pkl", "rb")); ev = json.load(open(F_ / "eval" / f"{i}.json")); rho = ev["fns"]["V"]["level"]
cfg = C.CertConfig(**cert["cfg"]); AN.enable_from_cert(cert); g = cfg.radius
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]], alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
cl = Vr.ClosedLoop(G_, d, int(s[1:]), v); T = jax.jit(cl.chart_map()); ell = C.make_ell(cl.G); V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
z = np.linspace(-g, g, 20_000_001)[:, None]; B = 1_000_000
cat = lambda f, x: np.concatenate([np.asarray(f(jp.asarray(x[k:k + B]))) for k in range(0, len(x), B)])
vz = cat(V, z); Fz = cat(T, z); vF = cat(V, Fz); l = cat(ell, z)
inr = vz <= rho; cont = inr & (np.abs(Fz[:, 0]) > g); dec = inr & (l >= cfg.eps_ball) & (vF - vz >= 0); bad = cont | dec
k = jax.random.split(jax.random.key(7), 8); X0 = cl.G.sample(k[2], 200000, 1.0); z0 = np.asarray(cl.G.Log(X0)); v0 = np.asarray(V(jp.asarray(z0))); inb = np.abs(z0[:, 0]) <= g
new = rho if not bad.any() else (1 - 1e-9) * float(vz[bad].min())
rec = dict(id=i, level=rho, n_grid=len(z), spacing=float(2 * g / (len(z) - 1)), n_in_region=int(inr.sum()),
           containment_violations=int(cont.sum()), decrease_violations=int(dec.sum()),
           violating_z=[float(bad_z) for bad_z in z[bad, 0][:20]], lowered_level=new,
           region_mu0=float((inb & (v0 <= rho)).mean()), region_mu0_lowered=float((inb & (v0 <= new)).mean()),
           region_mu0_eval=ev["fns"]["V"]["region_mu0"])
Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec))
