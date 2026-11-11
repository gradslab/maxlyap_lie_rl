"""Train the candidate (paper recipe, Algorithm 1) for a closed loop under a deployment variant, with the
ablation switches of lie_rl/rx_certificate.py. With no switch set it is scripts/train_certificate.py with
the _final5 environment (checked: identical parameters to certificate.train_certificate).

Usage: python scripts/rx_train_certificate.py GROUP DIR SEED VARIANT OUT_PICKLE [ITERS]
Recipe environment (as in paper_l4dc_v2/README.md): COST_POWER=2 ZUBOV_MODE=V W_DATA=0 BND=0 ZETA=0.01
  RADIUS_DIV=8 EPS_FIXED=1 RADIAL_FRAC=0.5 HIDDEN=64,64 MINE=1 MINE_SCOPED=0 MINE_EXIT=1 W_CE=5 W_EXIT=1
  EXIT_MARGIN=1.2 (these are also the defaults below).
Candidate switch: ANCHOR=r_a (anchored candidate, lie_rl/anchor.py).
Ablation switches: ITERS_MULT, REC_REL=1, W_REC, DATA_FIXED, N_DATA, DATA_MODE, EPS_AUTO=1, LAM_MAX, W_SHELL, W_SPEC, ROLLOUT_LEN, CERT_SEED.
"""
import os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, rx_certificate as RC, variants as Vr

group, d, seed, variant, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
iters = int(sys.argv[6]) if len(sys.argv) > 6 else (8000 if group.startswith("SO") else 4000)
iters = int(iters * float(os.environ.get("ITERS_MULT", "1")))
env = lambda k, dflt: os.environ.get(k, dflt)
mine_on = env("MINE", "1") != "0"
d0 = C.CertConfig()
cl = Vr.ClosedLoop(group, d, seed, variant); T = cl.chart_map()
gamma = float(np.pi) - float(np.pi) / float(env("RADIUS_DIV", "8"))
cfg = C.CertConfig(m=cl.m, radius=gamma, alpha_max=cl.amax, iters=iters, seed=int(env("CERT_SEED", str(seed))),
                   w_shell=float(env("W_SHELL", str(d0.w_shell))), w_data=float(env("W_DATA", "0")),
                   mine_scoped=env("MINE_SCOPED", "0") != "0", mine_keep=d0.mine_keep if mine_on else 0,
                   w_ce=float(env("W_CE", "5" if mine_on else "0")),
                   mine_exit=mine_on and env("MINE_EXIT", "1") == "1", w_exit=float(env("W_EXIT", "1")),
                   exit_margin=float(env("EXIT_MARGIN", "1.2")), hidden=tuple(int(x) for x in env("HIDDEN", "64,64").split(",")),
                   radial_frac=float(env("RADIAL_FRAC", "0.5")), cost_power=int(env("COST_POWER", "2")),
                   data_mode=env("DATA_MODE", "mse"), zubov_mode=env("ZUBOV_MODE", "V"), dec_mode="abs",
                   eps_fixed=float(env("EPS_FIXED", "1")), use_bnd=env("BND", "0") != "0",
                   zeta_train=float(env("ZETA", "0.01")), eps_ball=d0.eps_ball,
                   lam_max=float(env("LAM_MAX", str(d0.lam_max))), w_spec=float(env("W_SPEC", str(d0.w_spec))),
                   rollout_len=int(env("ROLLOUT_LEN", str(d0.rollout_len))))
knobs = dict(w_rec=float(env("W_REC", "1")), data_fixed=None if env("DATA_FIXED", "") == "" else float(env("DATA_FIXED", "")),
             n_data=int(env("N_DATA", "4096")), eps_auto=env("EPS_AUTO", "0") == "1", rec_rel=env("REC_REL", "0") == "1")
anchor = None
if env("ANCHOR", "") != "":
    from lie_rl import anchor as AN
    import json as _json, subprocess, tempfile
    _tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    subprocess.run([sys.executable, str(Path(__file__).parent / "rx_local_quadratic.py"), group, d, str(seed), variant, _tmp],
                   check=True, env={**os.environ, "JAX_ENABLE_X64": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    P_m = np.asarray(_json.load(open(_tmp))["P"])
    anchor = dict(P=np.asarray(P_m), r_a=float(env("ANCHOR", "0.3")))
    AN.enable(anchor["P"], anchor["r_a"])
print(f"=== rx certificate {group} {d} seed {seed} variant {variant}: iters {iters}, cfg {cfg._asdict()}, knobs {knobs}", flush=True)
t0 = time.time()
p, hist, cfg = RC.train_certificate(T, cl.G, cfg, **knobs)
ell = C.make_ell(cl.G)
z = C.sample_ball(jax.random.key(1234), 200000, cl.m, gamma)
gap = C.V_chart(p, cfg, ell, T(z)) - C.V_chart(p, cfg, ell, z)
mask = ell(z) >= cfg.eps_ball
frac = float(jp.mean(gap[mask] < 0))
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(dict(psi=[(np.asarray(w), np.asarray(b)) for w, b in p.psi], alpha_raw=float(p.alpha_raw),
                     eps_raw=float(p.eps_raw), cfg=cfg._asdict(), knobs=knobs, group=group, dir=d, seed=seed,
                     variant=variant, anchor=anchor, sampled_decrease_fraction=frac, wall_s=time.time() - t0), f)
print(f"    sampled decrease outside N_b: {100 * frac:.3f}%   ==> wrote {out} ({time.time() - t0:.0f}s)", flush=True)
