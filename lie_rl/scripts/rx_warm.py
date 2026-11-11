"""Recompute the candidate for a perturbed closed loop starting from the final candidate of the same policy without
the error. The anchor P_m is recomputed for the perturbed closed loop, the network psi starts
from the nominal candidate, and training runs ITERS iterations (default 1000, one refinement round) with the residual
weight at one from the start (ramp=False, as in refinement). Refinement (rx_refine.py) and evaluation (rx_eval.py)
then follow unchanged. Everything else (constants, knobs, r_a, seeds) is taken from the nominal candidate.

Usage: python scripts/rx_warm.py GROUP DIR SEED VARIANT OUT_PICKLE [ITERS]
"""
import json, os, pickle, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, rx_certificate as RC, variants as Vr
from lie_rl import anchor as AN

group, d, seed, variant, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
assert variant != "base"
iters = int(sys.argv[6]) if len(sys.argv) > 6 else int(os.environ.get("WARM_ITERS", "1000"))
base_id = f"{group}__{d}__s{seed}__base"
nom_path = Path(__file__).parents[1] / "results" / "final" / "cert" / f"{base_id}.pkl"
nom = pickle.load(open(nom_path, "rb"))
assert nom.get("anchor") is not None, "nominal candidate is not anchored"
cfg = C.CertConfig(**nom["cfg"])._replace(iters=iters)
knobs = {k_: v_ for k_, v_ in nom.get("knobs", {}).items() if k_ in ("w_rec", "data_fixed", "n_data", "eps_auto", "rec_rel")}
cl = Vr.ClosedLoop(group, d, seed, variant); T = cl.chart_map()
_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
subprocess.run([sys.executable, str(Path(__file__).parent / "rx_local_quadratic.py"), group, d, str(seed), variant, _tmp],
               check=True, env={**os.environ, "JAX_ENABLE_X64": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
P_m = np.asarray(json.load(open(_tmp))["P"])
anchor = dict(P=P_m, r_a=float(nom["anchor"]["r_a"]))
AN.enable(anchor["P"], anchor["r_a"])
p0 = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in nom["psi"]],
                  alpha_raw=jp.asarray(nom["alpha_raw"]), eps_raw=jp.asarray(nom["eps_raw"]))
print(f"=== warm start {group} {d} seed {seed} variant {variant} from {base_id}: iters {iters}", flush=True)
t0 = time.time()
p, hist, cfg = RC.train_certificate(T, cl.G, cfg, verbose=False, p_init=p0, ramp=False, **knobs)
ell = C.make_ell(cl.G)
z = C.sample_ball(jax.random.key(1234), 200000, cl.m, cfg.radius)
gap = C.V_chart(p, cfg, ell, T(z)) - C.V_chart(p, cfg, ell, z)
mask = ell(z) >= cfg.eps_ball
frac = float(jp.mean(gap[mask] < 0))
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(dict(psi=[(np.asarray(w), np.asarray(b)) for w, b in p.psi], alpha_raw=float(p.alpha_raw),
                     eps_raw=float(p.eps_raw), cfg=cfg._asdict(), knobs=nom.get("knobs", {}), group=group, dir=d, seed=seed,
                     variant=variant, anchor=anchor, sampled_decrease_fraction=frac, wall_s=time.time() - t0,
                     warm_from=str(nom_path), warm_iters=iters), f)
print(f"    sampled decrease outside N_b: {100 * frac:.3f}%   ==> wrote {out} ({time.time() - t0:.0f}s)", flush=True)
