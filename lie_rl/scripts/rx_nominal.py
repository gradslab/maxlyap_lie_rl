"""Does the certificate of the nominal closed loop still hold after an actuator error? (results/review/nominal)"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, variants as Vr
from lie_rl import anchor as AN

assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"
group, d, seed, variant, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
assert variant != "base"
N = int(os.environ.get("N_NOM", "1000000")); t_start = time.time()
base_id = f"{group}__{d}__s{seed}__base"
F_ = Path(__file__).parents[1] / "results" / "final"
cert = pickle.load(open(F_ / "cert" / f"{base_id}.pkl", "rb")); ev = json.load(open(F_ / "eval" / f"{base_id}.json"))
rho = float(ev["fns"]["V"]["level"])
rho = float(os.environ.get("NOMINAL_LEVEL", rho))
cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
ell = C.make_ell(G); V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))


def evalb(z, fn, bs=100000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])


z = np.asarray(C.sample_ball(jax.random.key(20261018), N, m, gamma))
v = evalb(z, V); Fz = evalb(z, Tj); vF = evalb(Fz, V); l = evalb(z, ell)
inreg = v <= rho; outnb = l >= r_b
dec_bad = inreg & outnb & (vF - v >= 0); cont_bad = inreg & (np.linalg.norm(Fz, axis=-1) > gamma)

k = jax.random.split(jax.random.key(7), 8)
X0 = G.sample(k[2], 200000, 1.0); z0 = np.asarray(G.Log(X0)); in0 = (np.linalg.norm(z0, axis=-1) <= gamma)
in0 = in0 & (evalb(z0, V) <= rho)
from lie_rl import roa as R
A_ = R.jacobian_at_zero(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A_))))
N_ROLL = 600 if spr <= 0.99 else int(min(20000, 600 + np.ceil(np.log(1e-3) / np.log(spr))))
roll = jax.jit(lambda Y: jax.lax.scan(lambda Xc, _: (cl.step(Xc), None), Y, None, length=N_ROLL)[0])
Xn = np.concatenate([np.asarray(roll(X0[i:i + 50000])) for i in range(0, X0.shape[0], 50000)])
th, tr = G.err_parts(jp.asarray(Xn)); conv = np.asarray((th < 0.05) & (tr < 0.05))
out = dict(id=f"{group}__{d}__s{seed}__{variant}", nominal=base_id, nominal_level=rho, n=N,
           n_in_nominal_region=int(inreg.sum()), n_in_region_outside_Nb=int((inreg & outnb).sum()),
           spr=spr, n_roll=N_ROLL, decrease_violations=int(dec_bad.sum()), containment_violations=int(cont_bad.sum()),
           violation_fraction_of_region=float((dec_bad | cont_bad).sum() / max(1, (inreg & outnb).sum())),
           nominal_region_mu0=float(in0.mean()), nonconverged_in_nominal_region_mu0=float((in0 & ~conv).mean()),
           wall_s=time.time() - t_start)
Path(out_path).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out))
