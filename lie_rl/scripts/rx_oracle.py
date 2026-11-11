"""Oracle region: the maximal Lyapunov function V_m itself under the level rule.

Usage: JAX_ENABLE_X64=1 python scripts/rx_oracle.py GROUP DIR SEED VARIANT OUT_JSON
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import certificate as C, roa as R, variants as Vr

group, d, seed, variant, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
t0 = time.time()
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
gamma = float(np.pi - np.pi / 8)
A = R.jacobian_at_zero(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A))))
N = int(min(20000, max(3000, 600 + np.ceil(np.log(1e-8) / np.log(max(spr, 1e-3))))))


@jax.jit
def vm_long(z):
    def body(c, _):
        X, acc = c
        return (cl.step(X), acc + jp.sum((G.obs(X) - G.obs(G.identity(1))) ** 2, axis=-1)), None
    (Xf, acc), _ = jax.lax.scan(body, (G.Exp(z), jp.zeros(z.shape[0])), None, length=N)
    th, tr = G.err_parts(Xf)
    return acc, jp.maximum(th, tr)


def vm_batched(z, bs=10000):
    outs = [vm_long(jp.asarray(z[i:i + bs])) for i in range(0, len(z), bs)]
    v = np.concatenate([np.asarray(o[0]) for o in outs]); f = np.concatenate([np.asarray(o[1]) for o in outs])
    return np.where(f < 0.05, v, np.inf)


k = jax.random.split(jax.random.key(7), 8)
Z1 = np.asarray(jp.concatenate([C.sample_ball(k[1], 400000, m, gamma), C.sample_ball_radial(k[7], 400000, m, gamma)], 0))
Z2 = np.asarray(C.sample_ball(jax.random.split(jax.random.key(99), 3)[2], 400000, m, gamma))
Z = np.concatenate([Z1, Z2]); sub = np.random.default_rng(0).choice(len(Z), 200000, replace=False); Zs = Z[sub]
FZ = np.concatenate([np.asarray(Tj(jp.asarray(Zs[i:i + 50000]))) for i in range(0, len(Zs), 50000)])
cont_bad = np.linalg.norm(FZ, axis=-1) > gamma
vmZ = vm_batched(Zs)
bad = cont_bad | ~np.isfinite(vmZ)
level = float(np.max(vmZ[np.isfinite(vmZ)]))
if bad.any(): level = min(level, (1 - 1e-9) * float(np.min(vmZ[bad])))
X0 = G.sample(k[2], 200000, 1.0); z0 = np.asarray(G.Log(X0)); in0 = np.linalg.norm(z0, axis=-1) <= gamma
vm0 = vm_batched(z0)
rec = dict(group=group, dir=d, seed=seed, variant=variant, spr=spr, N_rollout=N, level=level,
           n_containment_bad=int(cont_bad.sum()), n_nonconverged=int((~np.isfinite(vmZ)).sum()),
           region_mu0=float(np.mean(in0 & (vm0 <= level))), true_in_ball_mu0=float(np.mean(in0 & np.isfinite(vm0))),
           ball_mu0=float(in0.mean()), wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
print(json.dumps(rec))
