"""lambda_max(A_new^T P_old A_new - P_old) for the 14 kept candidates (P_m of the loop without the actuator error)."""
import json, glob, os, pickle, sys, time
from pathlib import Path
ROOT = Path(sys.argv[1]); OUT = sys.argv[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
import numpy as np, jax
from lie_rl import anchor as AN, certificate as C, variants as Vr
rows = []
for f in sorted(glob.glob(str(ROOT / "results/review/F/rloc/*.json"))):
    d = json.load(open(f))
    if d.get("P_m_matches_certificate", True): continue
    cl = Vr.ClosedLoop(d["group"], d["dir"], int(d["seed"]), d["variant"]); G = cl.G; m = cl.m; T = cl.chart_map()
    cert = pickle.load(open(d["cert"], "rb"))
    P_old = np.asarray(cert["anchor"]["P"], dtype=float)
    P_new, A, Qg = AN.local_quadratic(T, G, m)
    A = np.asarray(A, dtype=float); P_new = np.asarray(P_new, dtype=float); Qg = np.asarray(Qg, dtype=float)
    M = A.T @ P_old @ A - P_old
    lam = float(np.linalg.eigvalsh((M + M.T) / 2).max())
    chk = float(np.abs(A.T @ P_new @ A - P_new + Qg).max())
    rows.append(dict(group=d["group"], dir=d["dir"], seed=int(d["seed"]), variant=d["variant"], spr=float(np.abs(np.linalg.eigvals(A)).max()),
                     lam_max=lam, lam_min_Q=float(np.linalg.eigvalsh(Qg).min()), ratio_to_lamminQ=lam / float(np.linalg.eigvalsh(Qg).min()),
                     stein_residual_new=chk, P_old_vs_P_new_relfro=float(np.linalg.norm(P_old - P_new) / np.linalg.norm(P_new))))
    print(rows[-1], flush=True)
json.dump(dict(status="exact linear algebra; A by central differences as in anchor.local_quadratic", rows=rows), open(OUT, "w"), indent=1)
print("n =", len(rows), "all negative:", all(r["lam_max"] < 0 for r in rows))
