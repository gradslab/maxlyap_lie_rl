"""Tier 2: near-equilibrium quadratic patch (P6-style, transported)."""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import scipy.linalg

from . import so3


def linearize(T_chart):
    """A = dT/dz at z = 0 (requires the A5-corrected policy: T(0) = 0)."""
    A = jax.jacfwd(lambda z: T_chart(z[None])[0])(jp.zeros(3))
    return np.asarray(A)


def spectral_check(T_chart):
    """A6 gate: spectral radius of the chart linearization < 1."""
    A = linearize(T_chart)
    eigs = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigs))), A


def quadratic_patch(T_chart, r2: float = 0.35, n_est: int = 500_000, safety: float = 2.0, seed: int = 11):
    """Build (P, c_R, r_star): V_P decreases strictly on 0 < ||z|| <= r_star."""
    A = linearize(T_chart)
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    assert rho_A < 1.0, f"A6 fails: spectral radius {rho_A:.4f} >= 1"
    Q = np.eye(3)
    P = scipy.linalg.solve_discrete_lyapunov(A.T, Q)

    key = jax.random.key(seed)
    batch = 131_072
    c_R = 0.0
    done = 0
    Aj = jp.asarray(A)

    @jax.jit
    def rem_ratio(z):
        R = T_chart(z) - z @ Aj.T
        return jp.linalg.norm(R, axis=-1) / jp.linalg.norm(z, axis=-1) ** 2

    while done < n_est:
        key, k = jax.random.split(key)
        z = so3.sample_ball(k, batch, r2)
        c_R = max(c_R, float(jp.max(rem_ratio(z))))
        done += batch
    c_R *= safety

    lminQ = float(np.min(np.linalg.eigvalsh(Q)))
    nPA = float(np.linalg.norm(P @ A, 2))
    nP = float(np.linalg.norm(P, 2))
    coeffs = [nP * c_R**2, 2 * nPA * c_R, -lminQ]
    roots = np.roots(coeffs)
    r_pos = [float(r) for r in roots if np.isreal(r) and r > 0]
    r_star = min(min(r_pos) if r_pos else np.inf, r2)
    return {
        "A": A,
        "P": P,
        "rho_A": rho_A,
        "c_R": c_R,
        "r_star": float(r_star),
        "lminQ": lminQ,
    }


def make_patch_checker(P, actor_params, q_goal, step_len, T_chart, zeta_v=1e-9, obs_mode="matrix"):
    """A4(b): per-cell certified decrease of V_P(z) = z' P z."""
    import jax
    import jax.numpy as jp

    from .env import obs_matrix
    from .tier1a import _L_log, make_actor_local_lip

    Pj = jp.asarray(P)
    nP = float(np.linalg.norm(P, 2))
    sqrt2 = float(np.sqrt(2.0))
    actor_local_lip = make_actor_local_lip(actor_params)

    def VP(z):
        return jp.sum((z @ Pj) * z, axis=-1)

    @jax.jit
    def check(centers, h_inf):
        h2 = jp.sqrt(3.0) * h_inf
        zT = T_chart(centers)
        gc = VP(zT) - VP(centers)

        def one(c, zTc):
            from . import so3

            if obs_mode == "error_matrix":
                obs_c = so3.quat_to_mat(so3.exp_quat(c)).reshape(9)
            else:
                obs_c = obs_matrix(so3.quat_mul(q_goal, so3.exp_quat(c)), q_goal)
            L_act = actor_local_lip(obs_c, sqrt2 * h2)
            dF = 1.0 + sqrt2 * jp.pi * step_len * L_act
            th_img = jp.linalg.norm(zTc) + dF * h2
            L_T = _L_log(th_img) * dF
            r_img = L_T * h2
            inc_cell = nP * (2.0 * jp.linalg.norm(c) + h2) * h2
            inc_img = nP * (2.0 * jp.linalg.norm(zTc) + r_img) * r_img
            return inc_cell + inc_img

        slack = jax.vmap(one)(centers, zT)
        ub = gc + slack
        skipped = jp.zeros_like(ub, dtype=bool)
        return ub, ub <= -zeta_v, skipped

    return check
