"""Evaluation of one closed loop (policy and actuator error): level rule, regions of V, V_d and V_P, checks.

Usage: JAX_ENABLE_X64=1 python scripts/rx_eval.py GROUP DIR SEED VARIANT CERT_PICKLE OUT_JSON
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np, optax
from lie_rl import certificate as C, roa as R, variants as Vr

group, d, seed, variant, cert_path, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
BASELINES = os.environ.get("BASELINES", "1") != "0"
ADV_ROUNDS = int(os.environ.get("ADV_ROUNDS", "50"))
N_ADV, ADV_STEPS, ADV_EVERY, ETA = 32768, 1000, 25, 1e-9
t_start = time.time()

cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m
T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(cert_path, "rb"))
cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
from lie_rl import anchor as AN
ANCHORED = AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G)
fns = {"V": jax.jit(lambda z: C.V_chart(p, cfg, ell, z))}
A = R.jacobian_at_zero(T, m); spr = float(np.max(np.abs(np.linalg.eigvals(A))))
if BASELINES:
    fns["Vd"] = jax.jit(lambda z: jp.sum(z * z, axis=-1))
    if spr < 1.0:
        Pj = jp.asarray(R.stein(A)); fns["VP"] = jax.jit(lambda z: jp.einsum("ni,ij,nj->n", z, Pj, z))
    if cl.has_value_critic():
        fns["Vv"] = jax.jit(cl.value_candidate())
if cert.get("anchor") is not None:
    Pm = jp.asarray(np.asarray(cert["anchor"]["P"])); fns["VPm"] = jax.jit(lambda z: jp.einsum("ni,ij,nj->n", z, Pm, z))
FNS = [x for x in os.environ.get("FNS", "").split(",") if x]
fns = {k_: f_ for k_, f_ in fns.items() if (k_ in FNS if FNS else k_ != "VPm")}
assert fns, f"no function selected (FNS={FNS})"


def evalb(z, fn, bs=100000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])


def roll(X, n):
    Xf, _ = jax.lax.scan(lambda Xc, _: (cl.step(Xc), None), X, None, length=n); return Xf


def converged(X, tol=0.05):
    th, tr = G.err_parts(X); return np.asarray((th < tol) & (tr < tol))


N_ROLL = 600 if spr <= 0.99 else int(min(20000, 600 + np.ceil(np.log(1e-3) / np.log(spr))))


def conv_batched(X, bs=50000):
    rj = jax.jit(lambda Y: roll(Y, N_ROLL))
    return np.concatenate([converged(rj(X[i:i + bs])) for i in range(0, X.shape[0], bs)])


k = jax.random.split(jax.random.key(7), 8)
Z1 = np.asarray(jp.concatenate([C.sample_ball(k[1], 400000, m, gamma), C.sample_ball_radial(k[7], 400000, m, gamma)], 0))
Z2 = np.asarray(C.sample_ball(jax.random.split(jax.random.key(99), 3)[2], 400000, m, gamma))
geo = []
for Z in (Z1, Z2):
    FZ = evalb(Z, Tj); geo.append(dict(Z=Z, FZ=FZ, l=evalb(Z, ell), rF=np.linalg.norm(FZ, axis=-1)))

n_h = 200000
X0 = G.sample(k[2], n_h, 1.0); conv0 = conv_batched(X0)
z0 = np.asarray(G.Log(X0)); r0 = np.linalg.norm(z0, axis=-1); inball0 = r0 <= gamma
zb = np.asarray(C.sample_ball(k[3], 200000, m, gamma)); convb = conv_batched(G.Exp(jp.asarray(zb)))
zn = np.asarray(C.sample_ball(jax.random.key(99), 6_000_000 if m >= 6 else 2_000_000, m, 0.2))
zn = zn[evalb(zn, ell) < r_b][:100000]; FN = evalb(zn, Tj)
nb_conv = conv_batched(G.Exp(jp.asarray(zn)))


def adversarial(fn, rho, pool, rng):
    """Adam ascent on f(F z) - f(z) and on |F z| - gamma from points of the region, with a penalty for"""
    starts = pool[rng.choice(len(pool), min(N_ADV, len(pool)), replace=False)]
    scale = max(abs(rho), 1e-12)

    def feas(zq, penalty):
        if penalty:
            return (jax.nn.relu(fn(zq) - rho) / scale + jax.nn.relu(r_b - ell(zq))
                    + jax.nn.relu(jp.linalg.norm(zq, axis=-1) - gamma))
        return (fn(zq) <= rho) & (ell(zq) >= r_b) & (jp.linalg.norm(zq, axis=-1) <= gamma)
    best, n_found, worst = np.inf, 0, -np.inf
    for score in (lambda zq: fn(T(zq)) - fn(zq), lambda zq: jp.linalg.norm(T(zq), axis=-1) - gamma):
        obj = lambda zq, score=score: -jp.sum(score(zq) - 100.0 * feas(zq, True))
        opt = optax.adam(3e-3); z = jp.asarray(starts); st = opt.init(z)

        @jax.jit
        def upd(zq, st, obj=obj):
            g = jax.grad(obj)(zq); u, st = opt.update(g, st); return optax.apply_updates(zq, u), st
        for i in range(ADV_STEPS):
            z, st = upd(z, st)
            if (i + 1) % ADV_EVERY == 0:
                f = np.asarray(feas(z, False)); s = np.asarray(score(z))
                if f.any(): worst = max(worst, float(s[f].max()))
                hit = f & (s >= 0)
                if hit.any():
                    n_found += int(hit.sum()); best = min(best, float(np.asarray(fn(z))[hit].min()))
    return best, n_found, worst


res = {}
member0, memberb = {}, {}
for name, fn in fns.items():
    t0 = time.time()
    H = []
    for g in geo:
        fz = evalb(g["Z"], fn); fF = evalb(g["FZ"], fn)
        H.append(dict(f=fz, bad=((g["l"] >= r_b) & (fF - fz >= 0)) | (g["rF"] > gamma)))
    rho = float(H[0]["f"].max())
    for h in H:
        if h["bad"].any(): rho = min(rho, (1 - ETA) * float(h["f"][h["bad"]].min()))
    rho_sampled = rho
    rounds = []; rng = np.random.default_rng(seed + 17)
    for _ in range(ADV_ROUNDS):
        pool = np.concatenate([g["Z"][(h["f"] <= rho) & (g["l"] >= r_b)] for g, h in zip(geo, H)])
        if len(pool) == 0:
            rounds.append(dict(rho=rho, found=0, worst_score=None, empty_region=True)); break
        lvl, n, worst = adversarial(fn, rho, pool, rng)
        rounds.append(dict(rho=rho, found=n, worst_score=worst))
        if n == 0: break
        rho = (1 - ETA) * lvl
    f0 = evalb(z0, fn); fb = evalb(zb, fn)
    in0 = inball0 & (f0 <= rho); inb = fb <= rho
    member0[name], memberb[name] = in0, inb
    fN = evalb(zn, fn); fFN = evalb(FN, fn); nb_in = fN <= rho
    out_nb = geo[1]["l"] >= r_b
    res[name] = dict(frac_decrease_violation_ball=float(H[1]["bad"][out_nb].mean()),
                     level_sampled=rho_sampled, level=rho, adversarial_rounds=rounds,
                     adversarial_converged=bool(not rounds or rounds[-1]["found"] == 0),
                     lowered_by_search=bool(rho < rho_sampled),
                     region_mu0=float(in0.mean()), region_ball_fraction=float(inb.mean()),
                     false_inclusions_mu0=int(np.sum(in0 & ~conv0)), false_inclusions_ball=int(np.sum(inb & ~convb)),
                     entry_samples=int(nb_in.sum()), entry_violations=int(np.sum(nb_in & (fFN > rho))),
                     nb_nonconverged=int(np.sum(nb_in & ~nb_conv)), wall_s=time.time() - t0)
    print(name, json.dumps({k_: v_ for k_, v_ in res[name].items() if k_ != "adversarial_rounds"}), flush=True)


def union(names):
    names = [n for n in names if n in member0]
    if not names: return None
    u0 = np.any([member0[n] for n in names], axis=0); ub = np.any([memberb[n] for n in names], axis=0)
    return dict(members=names, region_mu0=float(u0.mean()), region_ball_fraction=float(ub.mean()),
                false_inclusions_mu0=int(np.sum(u0 & ~conv0)))


unions = {"V|Vd": union(["V", "Vd"]), "V|Vd|VP": union(["V", "Vd", "VP"]), "Vd|VP": union(["Vd", "VP"]),
          "Vd|VP|Vv": union(["Vd", "VP", "Vv"]), "all": union(list(fns))}

if "V" in fns:
    vm = jax.jit(C.make_vm(T, ell, cfg._replace(rollout_len=N_ROLL)))
    zf = np.asarray(C.sample_ball(k[4], 20000, m, gamma)); lf = evalb(zf, ell); outf = lf >= r_b
    Vf = evalb(zf, fns["V"])[outf]; vmf = evalb(zf, vm, 5000)[outf]; rel = np.abs(Vf - vmf) / np.maximum(vmf, 1e-12)
else:
    rel = np.array([np.nan])

rec = dict(group=group, dir=d, seed=seed, variant=variant, cert=str(cert_path), fns_evaluated=list(fns), gamma=gamma, eps_ball=r_b,
           spr=spr, passes_stability=bool(spr <= 0.99), n_roll=N_ROLL, adv_rounds_cap=ADV_ROUNDS, anchored=ANCHORED, anchor=float(C.alpha_of(p, cfg)),
           true_mu0=float(conv0.mean()), ball_mu0=float(inball0.mean()), true_in_ball_mu0=float((conv0 & inball0).mean()),
           true_ball_fraction=float(convb.mean()), nb_samples=int(len(zn)), nb_nonconverged_all=int(np.sum(~nb_conv)),
           fns=res, unions=unions,
           fit_median=None if "V" not in fns else float(np.median(rel)), fit_p90=None if "V" not in fns else float(np.percentile(rel, 90)),
           fit_p99=None if "V" not in fns else float(np.percentile(rel, 99)),
           cert_wall_s=cert.get("wall_s"), eval_wall_s=time.time() - t_start)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
print("wrote", out_path, "true_in_ball", rec["true_in_ball_mu0"], "unions", json.dumps(unions), f"{rec['eval_wall_s']:.0f}s")
