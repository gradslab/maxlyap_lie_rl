"""Additional measurements for one closed loop.

Usage: JAX_ENABLE_X64=1 python scripts/rx_extra.py GROUP DIR SEED VARIANT CERT_PICKLE EVAL_JSON OUT_JSON
"""
import json, math, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, env as E, roa as R, variants as Vr
from lie_rl import anchor as AN
from lie_rl.ppo import actor_mean

assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"
group, d, seed, variant, cert_path, eval_path, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
N_HELD = int(os.environ.get("N_HELD", "1000000"))
BETA = 1e-6
t_start = time.time()

cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m
T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(cert_path, "rb"))
cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
anchored = AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G)
V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
ev = json.load(open(eval_path)); rho = float(ev["fns"]["V"]["level"])


def evalb(z, fn, bs=100000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])


out = dict(group=group, dir=d, seed=seed, variant=variant, cert=str(cert_path), eval=str(eval_path), level=rho,
           anchored=anchored, region_mu0_eval=ev["fns"]["V"]["region_mu0"])

zh = np.asarray(C.sample_ball(jax.random.key(20261017), N_HELD, m, gamma))
vh = evalb(zh, V); Fh = evalb(zh, Tj); vFh = evalb(Fh, V); lh = evalb(zh, ell)
inreg = vh <= rho; outnb = lh >= r_b
dec_bad = inreg & outnb & (vFh - vh >= 0)
cont_bad = inreg & (np.linalg.norm(Fh, axis=-1) > gamma)
n_bad = int(np.sum(dec_bad | cont_bad))
out["heldout"] = dict(n=N_HELD, key=20261017, law="uniform in the ball", n_in_region=int(inreg.sum()),
                      n_in_region_outside_Nb=int((inreg & outnb).sum()), decrease_violations=int(dec_bad.sum()),
                      containment_violations=int(cont_bad.sum()), beta=BETA,
                      bound_ball_fraction=(math.log(1 / BETA) / N_HELD) if n_bad == 0 else None,
                      worst_relative_increase=float(np.max(((vFh - vh) / np.maximum(vh, 1e-300))[inreg & outnb])) if (inreg & outnb).any() else None)
print("heldout", json.dumps(out["heldout"]), flush=True)

if variant == "base":
    zo = np.asarray(C.sample_ball(jax.random.key(31), 100000, m, gamma)); Xo = G.Exp(jp.asarray(zo))
    raw = actor_mean(cl.af, E.obs(G, Xo))
    tau_raw = np.asarray(E.clip_tau(G, raw, cl.amax)); tau_cor = np.asarray(E.clip_tau(G, raw - cl.bias, cl.amax))
    dtau = np.linalg.norm(tau_raw - tau_cor, axis=-1)
    out["offset"] = dict(bias_norm=float(np.linalg.norm(np.asarray(cl.bias))), bias_max_abs=float(np.max(np.abs(np.asarray(cl.bias)))),
                         alpha_max=float(cl.amax), max_increment_change=float(dtau.max()),
                         median_increment_change=float(np.median(dtau)),
                         max_increment_change_over_alpha_max=float(dtau.max() / cl.amax))
    print("offset", json.dumps(out["offset"]), flush=True)

sprs = {}
for h in (1e-4, 1e-5, 1e-6, 1e-7):
    A = R.jacobian_at_zero(T, m, h=h); sprs[f"{h:g}"] = float(np.max(np.abs(np.linalg.eigvals(A))))
out["spr"] = dict(values=sprs, used=sprs["1e-06"], max_abs_deviation_from_used=float(max(abs(v - sprs["1e-06"]) for v in sprs.values())),
                  eval_json_spr=ev["spr"])
print("spr", json.dumps(out["spr"]), flush=True)

k = jax.random.split(jax.random.key(7), 8)
zf = np.asarray(C.sample_ball(k[4], 20000, m, gamma)); outf = evalb(zf, ell) >= r_b
Vf = evalb(zf, V)[outf]
fit = {}
for N in (600, 3000):
    vm = jax.jit(C.make_vm(T, ell, cfg._replace(rollout_len=N)))
    vmf = evalb(zf, vm, 5000)[outf]; fin = np.isfinite(vmf)
    rel = np.abs(Vf[fin] - vmf[fin]) / np.maximum(vmf[fin], 1e-12)
    fit[str(N)] = dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90)), p99=float(np.percentile(rel, 99)),
                       n=int(fin.sum()), n_nonfinite=int((~fin).sum()))
out["fit"] = fit; out["fit_eval_json_median"] = ev.get("fit_median")
print("fit", json.dumps(fit), flush=True)

zt = np.asarray(C.sample_ball(jax.random.key(5), 100000, m, gamma)); ztj = jp.asarray(zt)
V(ztj).block_until_ready(); t0 = time.time()
for _ in range(5): V(ztj).block_until_ready()
tV = (time.time() - t0) / (5 * len(zt))
timing = dict(V_s_per_point=tV)
for N in (600, 3000):
    vm = jax.jit(C.make_vm(T, ell, cfg._replace(rollout_len=N))); zs = ztj[:10000]
    vm(zs).block_until_ready(); t0 = time.time(); vm(zs).block_until_ready()
    timing[f"Vm{N}_s_per_point"] = (time.time() - t0) / 10000
timing["threads"] = os.environ.get("OMP_NUM_THREADS")
out["timing"] = timing
print("timing", json.dumps(timing), flush=True)

out["wall_s"] = time.time() - t_start
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(out_path, "w"), indent=1)
print("wrote", out_path, f"{out['wall_s']:.0f}s")
