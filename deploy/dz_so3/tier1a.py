"""STALE. DO NOT USE AND DO NOT CITE."""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from . import so3
from .certificate import CertConfig, CertParams, V_chart, alpha_of
from .nets import mlp_apply


def _opnorm_bound(A):
    fro = jp.sqrt(jp.sum(A**2))
    n1 = jp.max(jp.sum(jp.abs(A), axis=0))
    ninf = jp.max(jp.sum(jp.abs(A), axis=1))
    return jp.minimum(fro, jp.sqrt(n1 * ninf))


def _tanh_deriv_bound(c, r):
    closest = jp.maximum(0.0, jp.abs(c) - r)
    return 1.0 / jp.cosh(closest) ** 2


def _tanh_deriv_interval(pre_c, pre_r):
    """Interval of tanh' over [c-r, c+r]: max at the point closest to 0,"""
    d_hi = 1.0 / jp.cosh(jp.maximum(0.0, jp.abs(pre_c) - pre_r)) ** 2
    d_lo = 1.0 / jp.cosh(jp.abs(pre_c) + pre_r) ** 2
    return 0.5 * (d_hi + d_lo), 0.5 * (d_hi - d_lo), d_hi


def _ijac_mul(Jc, Jr, Ac, Ar):
    """Interval matrix product (Jc +- Jr) @ (Ac +- Ar)."""
    return Jc @ Ac, jp.abs(Jc) @ Ar + Jr @ jp.abs(Ac) + Jr @ Ar


def make_actor_local_lip(actor_params):
    """Per-cell certified Lipschitz of the tanh actor via an interval"""

    def actor_local_lip(obs_c, delta2):
        W, b = actor_params[0]
        pre_c = obs_c @ W + b
        pre_r = jp.sqrt(jp.sum(W**2, axis=0)) * delta2
        Dc, Dr, d_hi = _tanh_deriv_interval(pre_c, pre_r)
        Jc, Jr = W * Dc[None, :], jp.abs(W) * Dr[None, :]
        c = jp.tanh(pre_c)
        r = jp.minimum(pre_r, d_hi * pre_r)
        for W, b in actor_params[1:]:
            pre_c, pre_r = c @ W + b, r @ jp.abs(W)
            Dc, Dr, d_hi = _tanh_deriv_interval(pre_c, pre_r)
            Jc, Jr = _ijac_mul(Jc, Jr, W * Dc[None, :], jp.abs(W) * Dr[None, :])
            c = jp.tanh(pre_c)
            r = jp.minimum(pre_r, d_hi * pre_r)
        return _interval_opnorm(Jc, Jr)

    return actor_local_lip


def _interval_opnorm(Jc, Jr):
    """Sound sup of ||M||_2 over {M : |M - Jc| <= Jr elementwise}:"""
    sn_c = jp.linalg.norm(Jc, ord=2)
    return jp.minimum(sn_c + jp.sqrt(jp.sum(Jr**2)), _opnorm_bound(jp.abs(Jc) + Jr))


def make_psi_local_lip(psi_params):
    """Per-cell certified Lipschitz of the ReLU net psi via an interval"""

    def psi_local_lip(z_c, box_r):
        c, r = z_c, jp.full((3,), box_r)
        Jc, Jr = None, None
        n = len(psi_params)
        for i, (W, b) in enumerate(psi_params):
            c, r = c @ W + b, r @ jp.abs(W)
            if i < n - 1:
                on = (c - r >= 0).astype(c.dtype)
                off = (c + r <= 0).astype(c.dtype)
                Dc = jp.where(off > 0, 0.0, jp.where(on > 0, 1.0, 0.5))
                Dr = jp.where(off + on > 0, 0.0, 0.5)
                Ac, Ar = W * Dc[None, :], jp.abs(W) * Dr[None, :]
                Jc, Jr = (Ac, Ar) if Jc is None else _ijac_mul(Jc, Jr, Ac, Ar)
                lo, hi = jp.maximum(c - r, 0.0), jp.maximum(c + r, 0.0)
                c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
            else:
                Jc, Jr = (W, jp.zeros_like(W)) if Jc is None else _ijac_mul(
                    Jc, Jr, W, jp.zeros_like(W)
                )
        return _interval_opnorm(Jc, Jr)

    return psi_local_lip


def _sin_abs_max(th_lo, th_hi):
    contains_pi2 = (th_lo <= jp.pi / 2) & (th_hi >= jp.pi / 2)
    return jp.where(contains_pi2, 1.0, jp.maximum(jp.abs(jp.sin(th_lo)), jp.abs(jp.sin(th_hi))))


def _L_log(theta):
    t = jp.clip(theta, 1e-9, jp.pi - 1e-9)
    return (t / 2.0) / jp.sin(t / 2.0)


def make_increment_fn(p: CertParams, cfg: CertConfig):
    """inc(x, r2): certified bound of max_{||z-x||<=r2} V(z) - V(x) (and the"""
    psi0 = mlp_apply(p.psi, jp.zeros(3), jax.nn.relu)
    a = alpha_of(p, cfg)
    psi_local_lip = make_psi_local_lip(p.psi)

    def inc(x, r2):
        B = jp.linalg.norm(mlp_apply(p.psi, x, jax.nn.relu) - psi0)
        L = psi_local_lip(x, r2)
        th = jp.linalg.norm(x)
        s_bound = _sin_abs_max(jp.maximum(th - r2 * jp.sqrt(3.0), 0.0), th + r2 * jp.sqrt(3.0))
        if cfg.quad_only:
            return 4.0 * a**2 * s_bound * r2
        return L * r2 * (2.0 * B + L * r2) + 4.0 * a**2 * s_bound * r2

    return inc


def make_cell_checker(
    p, cfg, actor_params, q_goal, bias, step_len, T_chart, zeta_v, rho=None, obs_mode="matrix"
):
    """Jitted: (centers, h_inf) -> (upper bound of g over cell, passes, skipped)."""
    sqrt2 = float(np.sqrt(2.0))
    inc = make_increment_fn(p, cfg)
    actor_local_lip = make_actor_local_lip(actor_params)
    from .env import obs_matrix

    def obs_of(c):
        if obs_mode == "error_matrix":
            return so3.quat_to_mat(so3.exp_quat(c)).reshape(9)
        return obs_matrix(so3.quat_mul(q_goal, so3.exp_quat(c)), q_goal)

    @jax.jit
    def check(centers, h_inf):
        h2 = jp.sqrt(3.0) * h_inf
        zT = T_chart(centers)
        Vc = V_chart(p, cfg, centers)
        gc = V_chart(p, cfg, zT) - Vc

        def one(c, zTc):
            obs_c = obs_of(c)
            L_act = actor_local_lip(obs_c, sqrt2 * h2)
            dF = 1.0 + sqrt2 * jp.pi * step_len * L_act
            th_img_max = jp.linalg.norm(zTc) + dF * h2
            L_T = _L_log(th_img_max) * dF
            s = inc(c, h2) + inc(zTc, L_T * h2)
            s = jp.where(th_img_max >= jp.pi - 1e-2, jp.inf, s)
            return s, inc(c, h2)

        slack, inc_cell = jax.vmap(one)(centers, zT)
        slack = slack * (1.0 + 1e-9) + 1e-12
        inc_cell = inc_cell * (1.0 + 1e-9) + 1e-12
        ub = gc + slack
        if rho is None:
            skipped = jp.zeros_like(ub, dtype=bool)
        else:
            skipped = (Vc - inc_cell) > rho
        return ub, skipped | (ub <= -zeta_v), skipped

    return check


class CellReport(NamedTuple):
    certified: bool
    n_cells_certified: int
    n_cells_failed: int
    volume_certified: float
    volume_unresolved: float
    volume_total: float
    max_depth_hit: int
    wall_s: float
    failed_cells: np.ndarray
    stats: dict


def certify_annulus(
    check_fn,
    radius: float,
    eps_z: float,
    h0: float = 0.1,
    max_depth: int = 8,
    batch: int = 16384,
    cell_budget: int = 6_000_000,
    verbose: bool = True,
):
    """Budgeted branch-and-bound over the annulus eps_z <= ||z|| <= radius."""
    t0 = time.time()
    h_inf = h0
    axis = np.arange(-radius + h_inf, radius + h_inf, 2 * h_inf)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    cur = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    n_cert = 0
    vol_cert = 0.0
    depth = 0
    stats = {"evals": 0}
    failed_final = np.zeros((0, 3))
    h_failed = h_inf
    total_evals_budget = cell_budget

    while True:
        h2 = np.sqrt(3.0) * h_inf
        d = np.linalg.norm(cur, axis=-1)
        keep = (d - h2 <= radius) & (d + h2 >= eps_z)
        cur = cur[keep]
        if len(cur) == 0:
            break
        passed = np.zeros(len(cur), dtype=bool)
        skipped_mask = np.zeros(len(cur), dtype=bool)
        for i in range(0, len(cur), batch):
            chunk = cur[i : i + batch]
            n_chunk = len(chunk)
            pad = batch - n_chunk
            if pad > 0:
                chunk = np.concatenate([chunk, np.tile(chunk[-1:], (pad, 1))])
            out = check_fn(jp.asarray(chunk), h_inf)
            ub, ok = out[0], out[1]
            passed[i : i + batch] = np.asarray(ok)[:n_chunk]
            if len(out) > 2:
                skipped_mask[i : i + batch] = np.asarray(out[2])[:n_chunk]
            stats["evals"] += n_chunk
        stats["skipped"] = stats.get("skipped", 0) + int(skipped_mask.sum())
        n_pass = int(passed.sum())
        n_cert += n_pass
        vol_cert += n_pass * (2 * h_inf) ** 3
        fails = cur[~passed]
        if verbose:
            print(
                f"  depth {depth}: h={h_inf:.4f} cells={len(cur)} pass={n_pass} "
                f"fail={len(fails)} ({time.time() - t0:.1f}s)",
                flush=True,
            )
        if len(fails) == 0:
            failed_final = np.zeros((0, 3))
            h_failed = h_inf
            break
        next_count = len(fails) * 8
        if depth >= max_depth or stats["evals"] + next_count > total_evals_budget:
            failed_final = fails
            h_failed = h_inf
            break
        h_inf /= 2
        offs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
        )
        cur = (fails[:, None, :] + offs[None, :, :] * h_inf).reshape(-1, 3)
        depth += 1

    vol_total = 4.0 / 3.0 * np.pi * (radius**3 - eps_z**3)
    vol_unres = len(failed_final) * (2 * h_failed) ** 3
    return CellReport(
        certified=len(failed_final) == 0,
        n_cells_certified=n_cert,
        n_cells_failed=len(failed_final),
        volume_certified=float(vol_cert),
        volume_unresolved=float(vol_unres),
        volume_total=float(vol_total),
        max_depth_hit=depth,
        wall_s=time.time() - t0,
        failed_cells=np.concatenate(
            [failed_final, np.full((len(failed_final), 1), h_failed)], axis=-1
        )
        if len(failed_final)
        else np.zeros((0, 4)),
        stats=stats,
    )


def sound_rho(p, cfg, radius, alpha_max, h_inf=0.01, mu=1e-3, batch=65536, verbose=False):
    """Sound sublevel cut over the inflation shell using the tight per-cell"""
    inc = make_increment_fn(p, cfg)
    r_out = radius + alpha_max
    h2 = np.sqrt(3.0) * h_inf
    axis = np.arange(-r_out + h_inf, r_out + h_inf, 2 * h_inf)

    @jax.jit
    def lower(centers):
        V_c = V_chart(p, cfg, centers)
        slack = jax.vmap(lambda c: inc(c, jp.sqrt(3.0) * h_inf))(centers)
        return V_c - (slack * (1.0 + 1e-9) + 1e-12)

    lows = []
    for x in axis:
        Y, Z = np.meshgrid(axis, axis, indexing="ij")
        centers = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=-1)
        d = np.linalg.norm(centers, axis=-1)
        keep = (d + h2 >= radius) & (d - h2 <= r_out)
        centers = centers[keep]
        for i in range(0, len(centers), batch):
            chunk = centers[i : i + batch]
            lows.append(float(jp.min(lower(jp.asarray(chunk)))))
    v_min = min(lows)
    return v_min - mu, v_min
