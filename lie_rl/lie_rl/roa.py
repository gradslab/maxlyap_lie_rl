"""Certified regions of attraction on a general matrix Lie group."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import env as E


class ROA(NamedTuple):
    ok: bool
    reason: str
    m: int
    rho_A: float
    lam_min: float
    lam_max: float
    L: float
    L_ball: float
    r_star: float
    level: float
    r_inscribed: float
    vol_fraction: float
    clip_inactive: bool
    F0: float
    jac_uncertainty: float
    steps_to_contract: float
    level_sampled: float
    vol_fraction_sampled: float


def chart_map(agent_actor, bias, G, alpha_max, actor_mean):
    """z -> Log(Phi(Exp z)) for the repaired policy."""
    def F(z):
        X = G.Exp(z)
        a = actor_mean(agent_actor, E.obs(G, X)) - bias
        return G.Log(E.apply_action(G, X, a, alpha_max))
    return F


def jacobian_at_zero(F, m, h=1e-6):
    """Central differences. The Euclidean norms inside F are not differentiable"""
    A = np.zeros((m, m))
    for j in range(m):
        ej = jp.zeros((1, m)).at[0, j].set(h)
        A[:, j] = np.asarray(F(ej) - F(-ej))[0] / (2 * h)
    return A


def jacobian_with_uncertainty(F, m, h=1e-6):
    """The Jacobian and an estimate of its own error, from two step sizes."""
    A1 = jacobian_at_zero(F, m, h)
    A2 = jacobian_at_zero(F, m, h / 2)
    return A2, float(np.linalg.norm(A1 - A2, 2))


def stein(A):
    """Solve A^T P A - P = -I for P positive definite."""
    P = solve_discrete_lyapunov(np.asarray(A).T, np.eye(A.shape[0]))
    return 0.5 * (P + P.T)


def remainder_constant(F, A, m, radius, key, n=200000, shells=24):
    """Largest observed ||F(z) - A z|| / ||z||^2 over a ball of the given radius."""
    A_j = jp.asarray(A)
    worst = 0.0
    per = max(1, n // shells)
    for i in range(shells):
        r = radius * float(10 ** (-2.0 * i / max(1, shells - 1)))
        key, k1 = jax.random.split(key)
        d = jax.random.normal(k1, (per, m))
        d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
        z = d * r
        resid = F(z) - z @ A_j.T
        ratio = jp.linalg.norm(resid, axis=-1) / (r ** 2)
        worst = max(worst, float(jp.max(ratio)))
    return worst


def decrease_radius(P, A, L):
    """Largest r with 2||PA|| L r + lam_max(P) L^2 r^2 < 1."""
    if L <= 0:
        return np.inf
    b = 2.0 * np.linalg.norm(P @ A, 2) * L
    c = float(np.max(np.linalg.eigvalsh(P))) * L * L
    return float((-b + np.sqrt(b * b + 4 * c)) / (2 * c))


def clip_active_fraction(agent_actor, bias, G, alpha_max, actor_mean, radius, m, key, n=100000):
    """Fraction of sampled points in the ball at which the radial clip bites."""
    d = jax.random.normal(key, (n, m))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    u = jax.random.uniform(jax.random.split(key)[1], (n, 1)) ** (1.0 / m)
    z = d * (radius * u)
    X = G.Exp(z)
    a = actor_mean(agent_actor, E.obs(G, X)) - bias
    tau = jp.clip(a, -1.0, 1.0) * alpha_max
    return float(jp.mean(jp.linalg.norm(tau, axis=-1) > alpha_max * (1 + 1e-12)))


def largest_sampled_level(F, P, level0, m, key, gamma, n=40000, factors=40):
    """Largest multiple of level0 on whose boundary no sampled point violates"""
    Pj = jp.asarray(P)
    Pinv = np.linalg.inv(P)
    lam_min = float(np.linalg.eigvalsh(P)[0])
    cap = lam_min * gamma ** 2
    best = min(level0, cap)
    for i in range(factors):
        lev = level0 * float(1.4 ** i)
        if lev > cap:
            break
        Lc = np.linalg.cholesky(Pinv * lev)
        key, k = jax.random.split(key)
        d = jax.random.normal(k, (n, m))
        d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
        z = jp.asarray(np.asarray(d) @ Lc.T)
        Fz = F(z)
        dV = (jp.einsum('ij,jk,ik->i', Fz, Pj, Fz)
              - jp.einsum('ij,jk,ik->i', z, Pj, z))
        if float(jp.max(dV)) >= 0.0:
            break
        best = lev
    return best


def certify(agent_actor, bias, ep, actor_mean, key=None, R0=None, inflate=1.5):
    """Run the whole near-goal leg for one policy."""
    G = ep.G
    m = G.m
    key = key if key is not None else jax.random.key(0)
    F = chart_map(agent_actor, bias, G, ep.alpha_max, actor_mean)

    F0 = float(jp.linalg.norm(F(jp.zeros((1, m)))))
    A, dA = jacobian_with_uncertainty(F, m)
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    steps = float(np.inf) if rho_A >= 1 else float(-1.0 / np.log(max(rho_A, 1e-300)))
    if rho_A >= 1.0:
        return (ROA(False, "rho(A) >= 1, the goal is not locally attracting", m,
                    rho_A, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0,
                    False, F0, dA, steps, 0.0, 0.0), A, None)
    if 1.0 - rho_A <= 20.0 * dA:
        return (ROA(False,
                    "rho(A) is within the finite difference error of one",
                    m, rho_A, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0,
                    0.0, False, F0, dA, steps, 0.0, 0.0), A, None)

    P = stein(A)
    ev = np.linalg.eigvalsh(P)
    lam_min, lam_max = float(ev[0]), float(ev[-1])
    if lam_min <= 0:
        return (ROA(False, "the Stein solution is not positive definite", m,
                    rho_A, lam_min, lam_max, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0,
                    False, F0, dA, steps, 0.0, 0.0), A, P)

    R = R0 if R0 is not None else min(0.5 * ep.alpha_max, 0.2)
    for _ in range(8):
        key, k = jax.random.split(key)
        L = inflate * remainder_constant(F, A, m, R, k)
        r_star = decrease_radius(P, A, L)
        if r_star <= R:
            break
        R = min(2.0 * R, float(np.pi) - ep.alpha_max)
    level = lam_min * r_star ** 2
    r_ins = np.sqrt(level / lam_max)

    key, k = jax.random.split(key)
    frac = clip_active_fraction(agent_actor, bias, G, ep.alpha_max,
                                actor_mean, r_star, m, k)

    det_P = float(np.linalg.det(P))
    vol_frac = float(level ** (m / 2) / (np.sqrt(det_P) * ep.work_scale ** m))

    key, k = jax.random.split(key)
    gamma = float(np.pi) - ep.alpha_max
    lev_s = largest_sampled_level(F, P, level, m, k, gamma)
    vol_s = float(lev_s ** (m / 2) / (np.sqrt(det_P) * ep.work_scale ** m))

    return (ROA(True, "certified", m, rho_A, lam_min, lam_max, L, R, r_star,
                level, r_ins, vol_frac, frac == 0.0, F0, dA, steps,
                lev_s, vol_s), A, P)
