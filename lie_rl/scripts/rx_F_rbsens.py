"""Sensitivity of the candidate's region to the near-goal radius r_b, with no retraining.

Usage: JAX_ENABLE_X64=1 python scripts/rx_F_rbsens.py GROUP DIR SEED VARIANT EVAL_JSON R_B OUT_JSON
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import jax, jax.numpy as jp, numpy as np
from lie_rl import anchor as AN, certificate as C, roa as R, variants as Vr
import rx_F_common as F

group, d, seed, variant, eval_path, r_b, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], float(sys.argv[6]), sys.argv[7]
ADV_ROUNDS = F.env_int("ADV_ROUNDS", 50); ADV_N = F.env_int("ADV_N", 32768); ADV_STEPS = F.env_int("ADV_STEPS", 1000)
t0 = time.time()
ev = json.load(open(eval_path)); cert_path = F.cert_of(eval_path)
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(cert_path, "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius
AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G); fn = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
A = R.jacobian_at_zero(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A)))); N_ROLL = F.n_roll_of(spr)
geo, k = F.geometry(m, gamma, Tj, ell)
X0 = G.sample(k[2], 200000 // F.S, 1.0); conv0 = F.conv_batched(cl, X0, N_ROLL)
z0 = np.asarray(G.Log(X0)); inball0 = np.linalg.norm(z0, axis=-1) <= gamma
zb = np.asarray(C.sample_ball(k[3], 200000 // F.S, m, gamma)); convb = F.conv_batched(cl, G.Exp(jp.asarray(zb)), N_ROLL)
zn, FN = F.nb_sample(m, r_b, ell, Tj); nb_conv = F.conv_batched(cl, G.Exp(jp.asarray(zn)), N_ROLL)

lr = F.level_rule(fn, geo, r_b, T, ell, gamma, seed, ADV_ROUNDS, ADV_N, ADV_STEPS); rho = lr["level"]
in0 = inball0 & (F.evalb(z0, fn) <= rho); inb = F.evalb(zb, fn) <= rho
fN = F.evalb(zn, fn); fFN = F.evalb(FN, fn); nb_in = fN <= rho
res = dict(lr, region_mu0=float(in0.mean()), region_ball_fraction=float(inb.mean()),
           false_inclusions_mu0=int(np.sum(in0 & ~conv0)), false_inclusions_ball=int(np.sum(inb & ~convb)),
           entry_samples=int(nb_in.sum()), entry_violations=int(np.sum(nb_in & (fFN > rho))),
           nb_nonconverged=int(np.sum(nb_in & ~nb_conv)))
res["checks_pass"] = bool(res["entry_violations"] == 0 and res["nb_nonconverged"] == 0 and res["false_inclusions_mu0"] == 0
                          and res["false_inclusions_ball"] == 0 and res["adversarial_converged"])
rec = dict(group=group, dir=d, seed=seed, variant=variant, cert=str(cert_path), eval=str(eval_path), r_b=r_b, candidate_eps_ball=cfg.eps_ball,
           gamma=gamma, spr=spr, n_roll=N_ROLL, adv_rounds_cap=ADV_ROUNDS, adv_n=ADV_N, adv_steps=ADV_STEPS, sample_scale=F.S,
           nb_samples=int(len(zn)), V=res, reference_r_b_0_1=dict(region_mu0=ev["fns"]["V"]["region_mu0"], level=ev["fns"]["V"]["level"]),
           wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "adversarial_rounds"}), "r_b", r_b, "reference", rec["reference_r_b_0_1"], f"{rec['wall_s']:.0f}s")
