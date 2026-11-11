"""The near-goal decrease neighbourhood for a final candidate: how far from the goal the decrease of the
anchored candidate holds, and whether it covers N_b = {ell < r_b}.

Usage: JAX_ENABLE_X64=1 python scripts/rx_F_rloc.py GROUP DIR SEED VARIANT CERT_PICKLE OUT_JSON
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import jax, jax.numpy as jp, numpy as np, optax
from lie_rl import anchor as AN, certificate as C, roa as R, variants as Vr
import rx_F_common as F

group, d, seed, variant, cert_path, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
N_R = F.env_int("N_R", 120); N_SHELL = F.env_int("N_SHELL", 20000) // F.S; N_DIR = F.env_int("N_DIR", 20000) // F.S
ASC_N = F.env_int("ASC_N", 256); ASC_STEPS = F.env_int("ASC_STEPS", 200)
C_FRAC = float(os.environ.get("C_FRAC", "0.5"))
t0 = time.time()
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map()
cert = pickle.load(open(cert_path, "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
assert AN.enable_from_cert(cert), "not an anchored candidate"
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G)
V = lambda z: C.V_chart(p, cfg, ell, z)
P_m, A, Qg = AN.local_quadratic(T, G, m)
P_store = np.asarray(cert["anchor"]["P"])
c = C_FRAC * float(np.linalg.eigvalsh(Qg).min())
P = R.stein(A); Lc = np.linalg.cholesky(P); Lj, Lij = jp.asarray(Lc.T), jp.asarray(np.linalg.inv(Lc.T))

ratio = jax.jit(lambda z: (V(T(z)) - V(z)) / jp.sum(z * z, axis=-1))
DF = jax.jit(jax.vmap(jax.jacfwd(lambda z: T(z[None])[0])))
pnorm = jax.jit(lambda J: jp.linalg.norm(jp.einsum("ij,njk,kl->nil", Lj, J, Lij), ord=2, axis=(1, 2)))
contr = lambda z: pnorm(DF(z))


def dirs(key, n):
    if m == 1: return np.array([[1.0], [-1.0]])
    u = np.asarray(jax.random.normal(key, (n, m))); return u / np.linalg.norm(u, axis=-1, keepdims=True)


def make_ascent(fn):
    """Adam ascent of fn on a sphere (radius r passed at call time, so it compiles once); returns the largest value."""
    proj = lambda u, r: r * u / jp.linalg.norm(u, axis=-1, keepdims=True)
    opt = optax.adam(1e-2)

    @jax.jit
    def upd(u, st, r):
        up, st = opt.update(jax.grad(lambda v: -jp.sum(fn(proj(v, r))))(u), st); return optax.apply_updates(u, up), st
    val = jax.jit(lambda u, r: fn(proj(u, r)))

    def run(r, U):
        if m == 1: return -np.inf
        u = jp.asarray(U); st = opt.init(u); r = jp.asarray(r)
        for _ in range(ASC_STEPS): u, st = upd(u, st, r)
        return float(np.max(np.asarray(val(u, r))))
    return run


asc_ratio, asc_contr = make_ascent(ratio), make_ascent(contr)


radii = np.logspace(-4, np.log10(gamma), N_R)
if m == 1:
    radii = np.unique(np.concatenate([np.logspace(-4, np.log10(gamma), 100000), np.linspace(1e-4, gamma, 1000000 // F.S)]))
U = dirs(jax.random.key(3), N_SHELL)
shell = []
if m == 1:
    z = np.concatenate([radii[:, None], -radii[:, None]]); q = F.evalb(z, ratio).reshape(2, -1).max(0); k_ = F.evalb(z, contr, 5000).reshape(2, -1).max(0)
    shell = [dict(r=float(r), max_ratio=float(a), max_contraction=float(b), ascent_found_worse_ratio=False, ascent_found_worse_contraction=False)
             for r, a, b in zip(radii, q, k_)]
for i, r in enumerate(radii if m > 1 else []):
    z = r * U
    q = F.evalb(z, ratio); k_ = F.evalb(z, contr, 5000)
    worst_q = U[np.argsort(q)[-ASC_N:]]; worst_k = U[np.argsort(k_)[-min(ASC_N, 64):]]
    qa = asc_ratio(r, worst_q); ka = asc_contr(r, worst_k)
    shell.append(dict(r=float(r), max_ratio=float(max(q.max(), qa)), max_contraction=float(max(k_.max(), ka)),
                      ascent_found_worse_ratio=bool(qa > q.max()), ascent_found_worse_contraction=bool(ka > k_.max())))


def largest_radius(ok):
    r = 0.0
    for s, o in zip(shell, ok):
        if not o: break
        r = s["r"]
    return r


r_loc = largest_radius([s["max_ratio"] <= -c for s in shell])
r_dec = largest_radius([s["max_ratio"] < 0 for s in shell])
r_con = largest_radius([s["max_contraction"] < 1 for s in shell])
Ud = dirs(jax.random.key(5), N_DIR); lo = np.zeros(len(Ud)); hi = np.full(len(Ud), 0.5)
ellj = jax.jit(ell)
for _ in range(60):
    mid = 0.5 * (lo + hi); inside = np.asarray(ellj(jp.asarray(mid[:, None] * Ud))) < r_b
    lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
zb = hi[:, None] * Ud
r_Nb = float(hi.max()); a_Nb = float(np.max(np.einsum("ni,ij,nj->n", zb, P, zb)))
r_ell = float(np.sqrt(a_Nb / np.linalg.eigvalsh(P).min()))
rec = dict(group=group, dir=d, seed=seed, variant=variant, cert=str(cert_path), sample_scale=F.S, status="sampled (shells + Adam ascent); not a rigorous bound",
           spr=float(np.max(np.abs(np.linalg.eigvals(A)))), c=c, c_frac=C_FRAC, lambda_min_Q=float(np.linalg.eigvalsh(Qg).min()),
           P_m_matches_certificate=bool(np.allclose(P_m, P_store, rtol=1e-6, atol=1e-9)), r_b=r_b, gamma=gamma,
           r_loc=r_loc, r_dec=r_dec, r_con=r_con, r_Nb=r_Nb, a_Nb=a_Nb, r_ellipsoid_Nb=r_ell,
           Nb_in_B_rloc=bool(r_Nb <= r_loc), Nb_in_B_rdec=bool(r_Nb <= r_dec),
           Nb_closed_by_contraction=bool(r_ell <= r_con), n_shells=N_R, n_per_shell=int(len(U)), n_dirs_Nb=int(len(Ud)),
           shells=shell, wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
print(json.dumps({k_: v_ for k_, v_ in rec.items() if k_ != "shells"}))
