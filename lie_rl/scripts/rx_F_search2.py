"""A second, independent search of each reported region.

Usage: [KNOWN_POINTS=json] JAX_ENABLE_X64=1 python scripts/rx_F_search2.py GROUP DIR SEED VARIANT EVAL_JSON OUT_JSON
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import jax, jax.numpy as jp, numpy as np
from lie_rl import anchor as AN, certificate as C, roa as R, variants as Vr
import rx_F_common as F

group, d, seed, variant, eval_path, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
SEARCH = F.env_int("SEARCH", 1); ADV_ROUNDS = F.env_int("ADV_ROUNDS", 50); ADV_N = F.env_int("ADV_N", 32768); ADV_STEPS = F.env_int("ADV_STEPS", 1000)
t0 = time.time()
ev = json.load(open(eval_path)); cert_path = F.cert_of(eval_path)
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(cert_path, "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G)
A = R.jacobian_at_zero(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A)))); N_ROLL = F.n_roll_of(spr)
fns = {"V": jax.jit(lambda z: C.V_chart(p, cfg, ell, z)), "Vd": jax.jit(lambda z: jp.sum(z * z, axis=-1))}
if spr < 1.0:
    Pj = jp.asarray(R.stein(A)); fns["VP"] = jax.jit(lambda z: jp.einsum("ni,ij,nj->n", z, Pj, z))
if cert.get("anchor") is not None:
    Pm = jp.asarray(np.asarray(cert["anchor"]["P"])); fns["VPm"] = jax.jit(lambda z: jp.einsum("ni,ij,nj->n", z, Pm, z))
fns = {k_: f_ for k_, f_ in fns.items() if k_ in ev["fns"]}
assert fns, "no function of the eval json"
KP = os.environ.get("KNOWN_POINTS")
kz = np.asarray([q["z"] for q in json.load(open(KP))["inside"] if q["kind"] == "containment"]) if KP else np.zeros((0, m))
if len(kz):
    assert np.all(np.linalg.norm(kz, axis=-1) <= gamma) and np.all(np.linalg.norm(np.asarray(Tj(jp.asarray(kz))), axis=-1) > gamma)

geo, k = F.geometry(m, gamma, Tj, ell)
X0 = G.sample(k[2], 200000 // F.S, 1.0); conv0 = F.conv_batched(cl, X0, N_ROLL)
z0 = np.asarray(G.Log(X0)); inball0 = np.linalg.norm(z0, axis=-1) <= gamma
zb = np.asarray(C.sample_ball(k[3], 200000 // F.S, m, gamma)); convb = F.conv_batched(cl, G.Exp(jp.asarray(zb)), N_ROLL)
zn, FN = F.nb_sample(m, r_b, ell, Tj); nb_conv = F.conv_batched(cl, G.Exp(jp.asarray(zn)), N_ROLL)

res = {}
for name, fn in fns.items():
    t1 = time.time(); rho0 = ev["fns"][name]["level"]; rho = rho0
    fz = [F.evalb(g["Z"], fn) for g in geo]
    rounds = []; rng = np.random.default_rng(seed + 17 + 1000 * SEARCH)
    known = None
    if len(kz):
        fk = np.asarray(fn(jp.asarray(kz))); hit = fk <= rho
        known = dict(values=fk.tolist(), inside=int(hit.sum()))
        if hit.any(): rho = (1 - F.ETA) * float(fk[hit].min())
    for _ in range(ADV_ROUNDS):
        pool = np.concatenate([g["Z"][(f <= rho) & (g["l"] >= r_b)] for g, f in zip(geo, fz)])
        if len(pool) == 0:
            rounds.append(dict(rho=rho, found=0, worst_score=None, empty_region=True)); break
        lvl, n, worst = F.adversarial(fn, rho, pool, rng, T, ell, gamma, r_b, ADV_N, ADV_STEPS)
        rounds.append(dict(rho=rho, found=n, worst_score=worst, least_f_found=None if n == 0 else lvl))
        if n == 0: break
        rho = (1 - F.ETA) * lvl
    in0 = inball0 & (F.evalb(z0, fn) <= rho); inb = F.evalb(zb, fn) <= rho
    fN = F.evalb(zn, fn); fFN = F.evalb(FN, fn); nb_in = fN <= rho
    r = dict(level_reported=rho0, level=rho, lowered=bool(rho < rho0), known_points=known, adversarial_rounds=rounds,
             adversarial_converged=bool(not rounds or rounds[-1]["found"] == 0),
             region_mu0_reported=ev["fns"][name]["region_mu0"], region_mu0=float(in0.mean()), region_ball_fraction=float(inb.mean()),
             false_inclusions_mu0=int(np.sum(in0 & ~conv0)), false_inclusions_ball=int(np.sum(inb & ~convb)),
             entry_samples=int(nb_in.sum()), entry_violations=int(np.sum(nb_in & (fFN > rho))),
             nb_nonconverged=int(np.sum(nb_in & ~nb_conv)), wall_s=time.time() - t1)
    r["checks_pass"] = bool(r["entry_violations"] == 0 and r["nb_nonconverged"] == 0 and r["false_inclusions_mu0"] == 0
                            and r["false_inclusions_ball"] == 0 and r["adversarial_converged"])
    res[name] = r
    print(name, json.dumps({k_: v_ for k_, v_ in r.items() if k_ != "adversarial_rounds"}), flush=True)
rec = dict(group=group, dir=d, seed=seed, variant=variant, cert=str(cert_path), eval=str(eval_path), search=SEARCH,
           rng_seed=seed + 17 + 1000 * SEARCH, r_b=r_b, gamma=gamma, spr=spr, n_roll=N_ROLL, adv_rounds_cap=ADV_ROUNDS,
           adv_n=ADV_N, adv_steps=ADV_STEPS, sample_scale=F.S, fns=res, wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
