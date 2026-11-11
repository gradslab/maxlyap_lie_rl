"""P_m of the anchored candidate (lie_rl/anchor.py), computed in float64: A = DF(0), H = Hessian of l^2 at 0,
A^T P_m A - P_m = -H. Usage: JAX_ENABLE_X64=1 python scripts/rx_local_quadratic.py GROUP DIR SEED VARIANT OUT_JSON
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, numpy as np
assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"
from lie_rl import anchor as AN, variants as Vr
g, d, s, v, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
cl = Vr.ClosedLoop(g, d, s, v)
P, A, H = AN.local_quadratic(cl.chart_map(), cl.G, cl.m)
json.dump(dict(P=P.tolist(), A=A.tolist(), H=H.tolist(), spr=float(np.max(np.abs(np.linalg.eigvals(A)))),
               P_eig=np.linalg.eigvalsh(P).tolist()), open(out, "w"), indent=1)
print(json.dumps(dict(P_eig=np.linalg.eigvalsh(P).round(4).tolist(), H_diag=np.diag(H).round(4).tolist())))
