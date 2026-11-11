"""Closed loops with a final candidate, and the level rule of rx_eval.py with the near-goal radius r_b and the search
sizes as parameters."""
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax

from lie_rl import certificate as C

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ETA = 1e-9
S = int(os.environ.get("F_SCALE", "1"))


def cert_of(eval_json):
    """the candidate of an evaluation: always the eval json's own "cert" field (on D's carry loops it is the nominal"""
    import json
    c = Path(json.load(open(eval_json))["cert"])
    return c if c.is_absolute() else ROOT / c


EXCLUDED_VARIANTS = ("m50", "m60")


def final_loops():
    """every closed loop with a final candidate and its evaluation: (id, group, dir, seed, variant, cert, eval),"""
    out = [(f.stem, f) for f in sorted((RES / "final" / "eval").glob("*.json"))]
    for sub in ("pert36", "pert_r2", "pert_r3"):
        out += [(f.name[:-len("__final.json")], f) for f in sorted((RES / "review" / sub / "eval").glob("*__final.json"))]
    rows = []
    for i, e in out:
        G, d, s, v = i.split("__")
        if v in EXCLUDED_VARIANTS: continue
        c = cert_of(e)
        if c.exists():
            rows.append((i, G, d, int(s[1:]), v, c, e))
    return rows


def evalb(z, fn, bs=100000):
    return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)]) if len(z) else np.zeros(0)


def roll(cl, X, n):
    Xf, _ = jax.lax.scan(lambda Xc, _: (cl.step(Xc), None), X, None, length=n)
    return Xf


def converged(G, X, tol=0.05):
    th, tr = G.err_parts(X)
    return np.asarray((th < tol) & (tr < tol))


def conv_batched(cl, X, n_roll, bs=50000):
    rj = jax.jit(lambda Y: roll(cl, Y, n_roll))
    return np.concatenate([converged(cl.G, rj(X[i:i + bs])) for i in range(0, X.shape[0], bs)])


def n_roll_of(spr):
    return 600 if spr <= 0.99 else int(min(20000, 600 + np.ceil(np.log(1e-3) / np.log(spr))))


def geometry(m, gamma, Tj, ell):
    """Z1 (mixture law) and Z2 (volume-uniform) of rx_eval.py, with their images."""
    k = jax.random.split(jax.random.key(7), 8)
    Z1 = np.asarray(jp.concatenate([C.sample_ball(k[1], 400000 // S, m, gamma), C.sample_ball_radial(k[7], 400000 // S, m, gamma)], 0))
    Z2 = np.asarray(C.sample_ball(jax.random.split(jax.random.key(99), 3)[2], 400000 // S, m, gamma))
    geo = []
    for Z in (Z1, Z2):
        FZ = evalb(Z, Tj); geo.append(dict(Z=Z, FZ=FZ, l=evalb(Z, ell), rF=np.linalg.norm(FZ, axis=-1)))
    return geo, k


def nb_sample(m, r_b, ell, Tj, n_keep=100000):
    n_keep = n_keep // S
    """points of N_b = {ell < r_b}; for r_b = 0.1 exactly the sample of rx_eval.py (ball of radius 0.2)."""
    n = (6_000_000 if m >= 6 else 2_000_000) // S
    zn = np.asarray(C.sample_ball(jax.random.key(99), n, m, 2.0 * r_b))
    zn = zn[evalb(zn, ell) < r_b][:n_keep]
    return zn, evalb(zn, Tj)


def adversarial(fn, rho, pool, rng, T, ell, gamma, r_b, n_adv, steps, every=25):
    """rx_eval.adversarial with r_b and the search sizes as parameters."""
    starts = pool[rng.choice(len(pool), min(n_adv, len(pool)), replace=False)]
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
        for i in range(steps):
            z, st = upd(z, st)
            if (i + 1) % every == 0:
                f = np.asarray(feas(z, False)); s = np.asarray(score(z))
                if f.any(): worst = max(worst, float(s[f].max()))
                hit = f & (s >= 0)
                if hit.any():
                    n_found += int(hit.sum()); best = min(best, float(np.asarray(fn(z))[hit].min()))
    return best, n_found, worst


def level_rule(fn, geo, r_b, T, ell, gamma, seed, adv_rounds=50, n_adv=32768, adv_steps=1000, fgeo=None):
    """three-stage level rule of rx_eval.py: sampled decrease (outside N_b) and containment on Z1 and Z2, then"""
    H = []
    for j, g in enumerate(geo):
        fz, fF = (evalb(g["Z"], fn), evalb(g["FZ"], fn)) if fgeo is None else fgeo[j]
        H.append(dict(f=fz, bad=((g["l"] >= r_b) & (fF - fz >= 0)) | (g["rF"] > gamma)))
    rho = float(H[0]["f"].max())
    for h in H:
        if h["bad"].any(): rho = min(rho, (1 - ETA) * float(h["f"][h["bad"]].min()))
    rho_sampled = rho
    rounds = []; rng = np.random.default_rng(seed + 17)
    for _ in range(adv_rounds):
        pool = np.concatenate([g["Z"][(h["f"] <= rho) & (g["l"] >= r_b)] for g, h in zip(geo, H)])
        if len(pool) == 0:
            rounds.append(dict(rho=rho, found=0, worst_score=None, empty_region=True)); break
        lvl, n, worst = adversarial(fn, rho, pool, rng, T, ell, gamma, r_b, n_adv, adv_steps)
        rounds.append(dict(rho=rho, found=n, worst_score=worst))
        if n == 0: break
        rho = (1 - ETA) * lvl
    out_nb = geo[1]["l"] >= r_b
    return dict(level_sampled=rho_sampled, level=rho, adversarial_rounds=rounds,
                adversarial_converged=bool(not rounds or rounds[-1]["found"] == 0),
                lowered_by_search=bool(rho < rho_sampled),
                frac_decrease_violation_ball=float(H[1]["bad"][out_nb].mean()) if out_nb.any() else 0.0)


def env_int(k, d):
    return int(os.environ.get(k, str(d)))
