"""Tier 0: mass falsification (always reported; gates everything downstream)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jp
import numpy as np

from . import so3
from .certificate import CertConfig, CertParams, V_chart, ell_chart, pgd_counterexamples


def falsify_decrease(
    p: CertParams,
    cfg: CertConfig,
    T_chart,
    n_samples: int = 10_000_000,
    zeta_v: float = 1e-4,
    batch: int = 262_144,
    seed: int = 123,
    rho: float | None = None,
):
    """Sample the decrease condition on the annulus; PGD from uniform inits."""
    t0 = time.time()
    key = jax.random.key(seed)

    @jax.jit
    def margins(k):
        z = so3.sample_ball(k, batch, cfg.radius)
        V0 = V_chart(p, cfg, z)
        m = -(V_chart(p, cfg, T_chart(z)) - V0)
        ok_region = ell_chart(z) >= cfg.eps_ball
        return z, m, ok_region, V0

    n_batches = int(np.ceil(n_samples / batch))
    stats = {
        "in": {"n": 0, "viol": 0, "worst": np.inf, "qs": []},
        "out": {"n": 0, "viol": 0, "worst": np.inf, "qs": []},
    }
    worst_z_in = None
    for _ in range(n_batches):
        key, k = jax.random.split(key)
        z, m, ok, V0 = margins(k)
        m_np, ok_np, V_np, z_np = (np.asarray(x) for x in (m, ok, V0, z))
        inside_mask = ok_np & ((V_np <= rho) if rho is not None else np.ones_like(ok_np))
        for name, mask in (("in", inside_mask), ("out", ok_np & ~inside_mask)):
            mm = m_np[mask]
            if len(mm) == 0:
                continue
            s = stats[name]
            s["n"] += len(mm)
            s["viol"] += int((mm < zeta_v).sum())
            if mm.min() < s["worst"]:
                s["worst"] = float(mm.min())
                if name == "in":
                    worst_z_in = z_np[mask][np.argmin(mm)]
            s["qs"].append(np.quantile(mm, [0.0, 0.001, 0.01, 0.5]))

    key, kp = jax.random.split(key)
    r_min = _r_min_of_eps_ball(cfg)
    z_pgd, gaps = pgd_counterexamples(p, cfg, T_chart, kp, r_min=r_min)
    m_pgd = -np.asarray(gaps)
    ell_ok = np.asarray(ell_chart(z_pgd)) >= cfg.eps_ball
    if rho is not None:
        V_pgd = np.asarray(V_chart(p, cfg, z_pgd))
        pgd_in = m_pgd[ell_ok & (V_pgd <= rho)]
    else:
        pgd_in = m_pgd[ell_ok]
    pgd_min_in = float(pgd_in.min()) if len(pgd_in) else np.inf
    pgd_min_all = float(m_pgd[ell_ok].min()) if ell_ok.any() else np.inf

    def summarize(s):
        if not s["qs"]:
            return {"n_checked": 0}
        qs = np.stack(s["qs"])
        return {
            "n_checked": int(s["n"]),
            "n_violations": int(s["viol"]),
            "worst_margin_sampled": s["worst"],
            "margin_q0": float(qs[:, 0].min()),
            "margin_q001": float(np.median(qs[:, 1])),
            "margin_q01": float(np.median(qs[:, 2])),
            "margin_median": float(np.median(qs[:, 3])),
        }

    return {
        "rho": rho,
        "inside_Drho": summarize(stats["in"]),
        "outside_Drho": summarize(stats["out"]),
        "worst_z_inside": None if worst_z_in is None else worst_z_in.tolist(),
        "worst_margin_pgd_inside": pgd_min_in,
        "worst_margin_pgd_all": pgd_min_all,
        "zeta_v": zeta_v,
        "wall_s": time.time() - t0,
    }


def _r_min_of_eps_ball(cfg: CertConfig) -> float:
    """Chart radius corresponding to ell = eps_ball: ell = 2*sqrt(2)*sin(th/2)."""
    return float(2.0 * np.arcsin(cfg.eps_ball / (2.0 * np.sqrt(2.0))))


def invariance_mc(
    p: CertParams,
    cfg: CertConfig,
    T_chart,
    rho: float,
    n_samples: int = 1_000_000,
    batch: int = 262_144,
    seed: int = 321,
):
    """One-step invariance of D(rho): sample z in ball with V<=rho, check"""
    key = jax.random.key(seed)
    n_in = 0
    n_bad = 0

    @jax.jit
    def chk(k):
        z = so3.sample_ball(k, batch, cfg.radius)
        Vz = V_chart(p, cfg, z)
        zT = T_chart(z)
        VT = V_chart(p, cfg, zT)
        inside = Vz <= rho
        bad = inside & ((VT > rho) | (jp.linalg.norm(zT, axis=-1) > cfg.radius))
        return inside.sum(), bad.sum()

    done = 0
    while done < n_samples:
        key, k = jax.random.split(key)
        i, b = chk(k)
        n_in += int(i)
        n_bad += int(b)
        done += batch
    return {"n_inside": n_in, "n_invariance_violations": n_bad}


def rho_estimate(
    p: CertParams, cfg: CertConfig, alpha_max: float, mu: float = 1e-3, n: int = 2_000_000, seed: int = 7
):
    """Tier-0 estimate of rho: dense sampling + local refinement of min V over"""
    key = jax.random.key(seed)
    r_in, r_out = cfg.radius, cfg.radius + alpha_max
    vmin = np.inf
    zmin = None
    batch = 262_144
    done = 0
    while done < n:
        key, k = jax.random.split(key)
        z = so3.sample_ball(k, batch, r_out, r_min=r_in)
        V = np.asarray(V_chart(p, cfg, z))
        i = int(np.argmin(V))
        if V[i] < vmin:
            vmin = float(V[i])
            zmin = z[i]
        done += batch
    zz = jp.asarray(zmin)
    gfun = jax.jit(jax.grad(lambda z: V_chart(p, cfg, z[None])[0]))
    for _ in range(200):
        zz = zz - 0.01 * gfun(zz)
        nrm = jp.linalg.norm(zz)
        nrm_c = jp.clip(nrm, r_in, r_out)
        zz = zz / nrm * nrm_c
    v_ref = float(V_chart(p, cfg, zz[None])[0])
    vmin = min(vmin, v_ref)
    return vmin - mu, vmin
