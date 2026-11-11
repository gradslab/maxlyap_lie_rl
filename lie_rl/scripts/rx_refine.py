"""Search-in-the-loop refinement of a trained candidate (counterexample-guided, as in neural Lyapunov training).

Usage: JAX_ENABLE_X64=1 python scripts/rx_refine.py GROUP DIR SEED VARIANT CERT_IN CERT_OUT
Env: ROUNDS (6), ITERS_PER (1000), N_COPIES (8), JITTER (0.05, relative to |z|)
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import gc
import jax, jax.numpy as jp, numpy as np, optax
from lie_rl import certificate as C, rx_certificate as RC, variants as Vr

group, d, seed, variant, cert_in, cert_out = sys.argv[1:7]; seed = int(seed)
ROUNDS = int(os.environ.get("ROUNDS", "6")); ITERS_PER = int(os.environ.get("ITERS_PER", "1000"))
N_COPIES = int(os.environ.get("N_COPIES", "8")); JITTER = float(os.environ.get("JITTER", "0.05"))
t0 = time.time()
cl = Vr.ClosedLoop(group, d, seed, variant); G = cl.G; m = cl.m; T = cl.chart_map(); Tj = jax.jit(T)
cert = pickle.load(open(cert_in, "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
KNOBS = {k_: v_ for k_, v_ in cert.get("knobs", {}).items() if k_ in ("w_rec", "data_fixed", "n_data", "eps_auto", "rec_rel")} if os.environ.get("REFINE_VARIANT_LOSS", "1") == "1" else {}
from lie_rl import anchor as AN
AN.enable_from_cert(cert)
p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                 alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
ell = C.make_ell(G); lj = jax.jit(ell)
X0 = G.sample(jax.random.key(4242), 50000, 1.0); z0 = np.asarray(G.Log(X0)); in0 = np.linalg.norm(z0, axis=-1) <= gamma


def evalb(z, fn, bs=100000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)]) if len(z) else np.zeros(0)


def assess(p, rnd):
    """sampled level, search-found violations, estimated region (fresh samples of round rnd)."""
    V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
    k = jax.random.split(jax.random.key(1000 + rnd), 2)
    Z = np.concatenate([np.asarray(C.sample_ball(k[0], 200000, m, gamma)), np.asarray(C.sample_ball_radial(k[1], 200000, m, gamma))])
    VZ = evalb(Z, V); FZ = evalb(Z, Tj); VF = evalb(FZ, V); lZ = evalb(Z, lj); rF = np.linalg.norm(FZ, axis=-1)
    dec_bad = (lZ >= r_b) & (VF - VZ >= 0); cont_bad = rF > gamma
    vmax = float(VZ[:200000].max()); level = vmax
    if (dec_bad | cont_bad).any(): level = min(level, (1 - 1e-9) * float(VZ[dec_bad | cont_bad].min()))
    found_dec, found_cont = [Z[dec_bad]], [Z[cont_bad]]
    level_sampled = level
    pool = Z[(VZ <= level) & (lZ >= r_b)]
    if len(pool):
        starts = pool[np.random.default_rng(rnd).choice(len(pool), min(32768, len(pool)), replace=False)]
        scale = max(abs(level), 1e-12)
        def feas(zq, penalty):
            if penalty:
                return jax.nn.relu(V(zq) - level) / scale + jax.nn.relu(r_b - ell(zq)) + jax.nn.relu(jp.linalg.norm(zq, axis=-1) - gamma)
            return (V(zq) <= level) & (ell(zq) >= r_b) & (jp.linalg.norm(zq, axis=-1) <= gamma)
        for which, score in (("dec", lambda zq: V(T(zq)) - V(zq)), ("cont", lambda zq: jp.linalg.norm(T(zq), axis=-1) - gamma)):
            obj = lambda zq, score=score: -jp.sum(score(zq) - 100.0 * feas(zq, True))
            opt = optax.adam(3e-3); z = jp.asarray(starts); st = opt.init(z)
            @jax.jit
            def upd(zq, st, obj=obj):
                g = jax.grad(obj)(zq); u, st = opt.update(g, st); return optax.apply_updates(zq, u), st
            for i in range(1000):
                z, st = upd(z, st)
                if (i + 1) % 25 == 0:
                    hit = np.asarray(feas(z, False)) & (np.asarray(score(z)) >= 0)
                    if hit.any():
                        (found_dec if which == "dec" else found_cont).append(np.asarray(z)[hit])
                        level = min(level, (1 - 1e-9) * float(np.asarray(V(z))[hit].min()))
    fd = np.unique(np.round(np.concatenate(found_dec), 7), axis=0); fc = np.unique(np.round(np.concatenate(found_cont), 7), axis=0)
    region = float(np.mean(in0 & (evalb(z0, V) <= level)))
    return dict(level=level, level_sampled=level_sampled, search_lowered=bool(level < level_sampled), vmax=vmax, region=region, n_dec=int(len(fd)), n_cont=int(len(fc)),
                n_dec_below_level_sampled=int(np.sum(dec_bad & (VZ <= vmax)))), fd, fc


rng = np.random.default_rng(seed)
pin_dec = np.zeros((0, m)); pin_cont = np.zeros((0, m))
best = None; log = []
for rnd in range(ROUNDS + 1):
    info, fd, fc = assess(p, rnd)
    info.update(round=rnd, wall_s=time.time() - t0); log.append(info); print(json.dumps(info), flush=True)
    if best is None or info["region"] > best[0]["region"]:
        best = (info, p)
    if (info["n_dec_below_level_sampled"] == 0 and not info["search_lowered"]) or rnd == ROUNDS:
        break
    if len(fd):
        Vfd = evalb(fd, jax.jit(lambda z: C.V_chart(p, cfg, ell, z))); fd = fd[np.argsort(Vfd)[:4000]]
        jit_ = fd[None] * (1 + JITTER * rng.standard_normal((N_COPIES, *fd.shape))) + 0.01 * rng.standard_normal((N_COPIES, *fd.shape)) * np.linalg.norm(fd, axis=-1, keepdims=True)
        new = np.concatenate([fd, jit_.reshape(-1, m)])
        new = new[(evalb(new, lj) >= r_b) & (np.linalg.norm(new, axis=-1) <= gamma)]
        pin_dec = np.concatenate([pin_dec, new])[-40000:]
    if len(fc):
        pin_cont = np.concatenate([pin_cont, fc[:4000]])[-20000:]
    p, _, _ = RC.train_certificate(T, G, cfg._replace(iters=ITERS_PER, seed=cfg.seed + 1000 + rnd), verbose=False, **KNOBS,
                                    p_init=p, ce_pinned=jp.asarray(pin_dec) if len(pin_dec) else None,
                                    exit_pinned=jp.asarray(pin_cont) if len(pin_cont) else None, ramp=False)
    jax.clear_caches(); gc.collect()
info, pbest = best
Path(cert_out).parent.mkdir(parents=True, exist_ok=True)
out = dict(cert); out.update(psi=[(np.asarray(w), np.asarray(b)) for w, b in pbest.psi], alpha_raw=float(pbest.alpha_raw),
                            eps_raw=float(pbest.eps_raw), refine_log=log, refine_best_round=info["round"],
                            refine_settings=dict(rounds=ROUNDS, iters_per=ITERS_PER, n_copies=N_COPIES, jitter=JITTER),
                            wall_s=float(cert.get("wall_s", 0)) + time.time() - t0)
pickle.dump(out, open(cert_out, "wb"))
print("best round", info["round"], "estimated region", info["region"], "wrote", cert_out, f"({time.time() - t0:.0f}s)")
