"""Rollout ground truth: Monte-Carlo ROA, V_m sums, coverage metric."""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from . import so3


def make_vm_fn(F_quat, horizon: int = 300, tol: float = 0.05, window: int = 10):
    """Returns vm_fn(z)->(Vm, converged) evaluated by vmapped rollouts."""

    @jax.jit
    def rollout(z):
        q0 = so3.exp_quat(z)

        def body(carry, _):
            q, vm = carry
            vm = vm + so3.ell(q)
            q2 = F_quat(q)
            return (q2, vm), so3.angle_of(q2)

        (qf, vm), angles = jax.lax.scan(body, (q0, jp.zeros(z.shape[0])), None, length=horizon)
        tail = angles[-window:]
        converged = jp.max(tail, axis=0) < tol
        return vm, converged

    def vm_fn(z):
        vm, _ = rollout(z)
        return vm

    def roa_fn(z):
        _, conv = rollout(z)
        return conv

    return vm_fn, roa_fn, rollout


def coverage(p, cfg, roa_fn, rho: float, n: int = 500_000, seed: int = 5):
    """P1 coverage: |{z in MC-ROA : V(z) <= rho}| / |MC-ROA| over the working"""
    from .certificate import V_chart

    key = jax.random.key(seed)
    batch = 65_536
    n_roa = 0
    n_roa_cert = 0
    n_tot = 0
    n_cert = 0
    while n_tot < n:
        key, k = jax.random.split(key)
        z = so3.sample_ball(k, batch, cfg.radius)
        conv = np.asarray(roa_fn(z))
        V = np.asarray(V_chart(p, cfg, z))
        cert = V <= rho
        n_tot += batch
        n_cert += int(cert.sum())
        n_roa += int(conv.sum())
        n_roa_cert += int((conv & cert).sum())
    return {
        "mc_roa_fraction": n_roa / n_tot,
        "certified_fraction_of_ball": n_cert / n_tot,
        "coverage_of_roa": (n_roa_cert / n_roa) if n_roa else float("nan"),
        "certified_outside_roa": n_cert - n_roa_cert,
        "n": n_tot,
    }
