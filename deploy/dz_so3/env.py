"""JAX reimplementation of so3_primer's idealized rotation task (E1)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from . import so3


class EnvParams(NamedTuple):
    step_len: float = 0.1
    tol: float = 0.1
    max_steps: int = 50
    reset_cone: float = 0.0

    @property
    def alpha_max(self) -> float:
        return float(jp.pi) * self.step_len


class EnvState(NamedTuple):
    q: jax.Array
    q_goal: jax.Array
    t: jax.Array
    key: jax.Array


def obs_matrix(q: jax.Array, q_goal: jax.Array) -> jax.Array:
    """concat(vec(R), vec(R_g)) row-major, 18-dim (facts 8b-C6)."""
    r = so3.quat_to_mat(q).reshape(q.shape[:-1] + (9,))
    g = so3.quat_to_mat(q_goal).reshape(q_goal.shape[:-1] + (9,))
    return jp.concatenate([r, g], axis=-1)


def obs_error_matrix(q: jax.Array, q_goal: jax.Array) -> jax.Array:
    """Amendment A7: obs = vec(R_g^T R) = vec(E), 9-dim, no goal channel."""
    qE = so3.quat_mul(so3.quat_conj(q_goal), q)
    return so3.quat_to_mat(qE).reshape(q.shape[:-1] + (9,))


OBS_FNS = {"matrix": obs_matrix, "error_matrix": obs_error_matrix}
OBS_DIMS = {"matrix": 18, "error_matrix": 9}


def apply_action(q: jax.Array, action: jax.Array, step_len: float) -> jax.Array:
    """One dynamics step: rel_scale tangent action + geodesic clamp."""
    alpha_max = jp.pi * step_len
    tau = jp.clip(action, -1.0, 1.0) * jp.pi * step_len
    tn = jp.linalg.norm(tau, axis=-1, keepdims=True)
    tau = tau * jp.minimum(1.0, alpha_max / jp.clip(tn, 1e-12))
    return so3.quat_mul(q, so3.exp_quat(tau))


def _sample_start(key, n, q_goal, reset_cone):
    """q_goal-Haar starts (reset_cone=0) or starts within the cone (error angle"""
    if reset_cone <= 0.0:
        return so3.sample_quat_uniform(key, n)
    k1, k2 = jax.random.split(key)
    ax = jax.random.normal(k1, (n, 3))
    ax = ax / jp.linalg.norm(ax, axis=-1, keepdims=True)
    th = reset_cone * jax.random.uniform(k2, (n, 1)) ** (1 / 3)
    return so3.quat_mul(q_goal, so3.exp_quat(ax * th))


def reset(key: jax.Array, n: int, params: EnvParams) -> EnvState:
    k1, k2, k3 = jax.random.split(key, 3)
    q_goal = so3.sample_quat_uniform(k2, n)
    return EnvState(
        q=_sample_start(k1, n, q_goal, params.reset_cone),
        q_goal=q_goal,
        t=jp.zeros(n, dtype=jp.int32),
        key=k3,
    )


def step(state: EnvState, action: jax.Array, params: EnvParams):
    """Vectorized env step with auto-reset after max_steps (truncation only)."""
    q_next = apply_action(state.q, action, params.step_len)
    t_next = state.t + 1
    angle = so3.d_geo(q_next, state.q_goal)
    reward = -angle / jp.pi
    truncated = t_next >= params.max_steps

    k1, k2, k3 = jax.random.split(state.key, 3)
    n = state.q.shape[0]
    g_rnd = so3.sample_quat_uniform(k2, n)
    q_rnd = _sample_start(k1, n, g_rnd, params.reset_cone)
    mask = truncated[:, None]
    new_state = EnvState(
        q=jp.where(mask, q_rnd, q_next),
        q_goal=jp.where(mask, g_rnd, state.q_goal),
        t=jp.where(truncated, 0, t_next),
        key=k3,
    )
    return new_state, q_next, reward, truncated


def make_closed_loop(
    policy_fn,
    params,
    q_goal: jax.Array,
    step_len: float,
    bias: jax.Array | None = None,
    obs_mode: str = "matrix",
):
    """Return F on error quaternions and its chart version T, at a fixed goal."""

    def F_quat(qE: jax.Array) -> jax.Array:
        if obs_mode == "error_matrix":
            o = so3.quat_to_mat(qE).reshape(qE.shape[:-1] + (9,))
        else:
            qR = so3.quat_mul(q_goal, qE)
            o = obs_matrix(qR, jp.broadcast_to(q_goal, qR.shape))
        a = policy_fn(params, o)
        if bias is not None:
            a = a - bias
        return so3.quat_mul(qE, so3.exp_quat(_clamped_tau(a, step_len)))

    def T_chart(z: jax.Array) -> jax.Array:
        return so3.log_quat(F_quat(so3.exp_quat(z)))

    return F_quat, T_chart


def _clamped_tau(a: jax.Array, step_len: float) -> jax.Array:
    alpha_max = jp.pi * step_len
    tau = jp.clip(a, -1.0, 1.0) * jp.pi * step_len
    tn = jp.linalg.norm(tau, axis=-1, keepdims=True)
    return tau * jp.minimum(1.0, alpha_max / jp.clip(tn, 1e-12))
