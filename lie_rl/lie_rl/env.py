"""The idealized goal-reaching task on a general matrix Lie group."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from .groups import GROUPS, Group


class EnvParams(NamedTuple):
    group: str = "SO2"
    step_len: float = 0.1
    T_max: float = 1.0
    max_steps: int = 50
    tol_rot: float = 0.1
    tol_trans: float = 0.05
    reward_shape: str = "linear"

    @property
    def G(self) -> Group:
        return GROUPS[self.group]

    @property
    def work_scale(self) -> float:
        return self.G.work_scale(self.T_max)

    @property
    def alpha_max(self) -> float:
        return self.work_scale * self.step_len


class EnvState(NamedTuple):
    E: jax.Array
    t: jax.Array
    key: jax.Array


def clip_tau(G: Group, action: jax.Array, alpha_max: float) -> jax.Array:
    """tau = radial_clip(clip(a, -1, 1) * alpha_max) (Definition 3.5)."""
    tau = jp.clip(action, -1.0, 1.0) * alpha_max
    tn = jp.linalg.norm(tau, axis=-1, keepdims=True)
    return tau * jp.minimum(1.0, alpha_max / jp.clip(tn, 1e-12))


def apply_action(G: Group, E: jax.Array, action: jax.Array, alpha_max: float):
    return G.compose(E, G.Exp(clip_tau(G, action, alpha_max)))


def cost(G: Group, E: jax.Array) -> jax.Array:
    """dist(E, e) = ||Log E||."""
    return jp.linalg.norm(G.Log(E), axis=-1)


def shaped_cost(G: Group, E: jax.Array, params: "EnvParams") -> jax.Array:
    """The cost the reward is built from."""
    d = cost(G, E)
    if params.reward_shape == "linear":
        return d
    delta = params.alpha_max
    return jp.where(d > delta, d, d * d / (2 * delta) + delta / 2)


def obs(G: Group, E: jax.Array) -> jax.Array:
    return G.obs(E)


def reset(key: jax.Array, n: int, params: EnvParams) -> EnvState:
    k1, k2 = jax.random.split(key)
    return EnvState(E=params.G.sample(k1, n, params.T_max),
                    t=jp.zeros(n, dtype=jp.int32), key=k2)


def step(state: EnvState, action: jax.Array, params: EnvParams):
    """Vectorized step with auto-reset on truncation (no termination)."""
    G = params.G
    E_next = apply_action(G, state.E, action, params.alpha_max)
    t_next = state.t + 1
    reward = -shaped_cost(G, E_next, params) / params.work_scale
    truncated = t_next >= params.max_steps

    k1, k2 = jax.random.split(state.key)
    n = state.E.shape[0]
    E_rnd = G.sample(k1, n, params.T_max)
    mask = truncated[:, None]
    new_state = EnvState(E=jp.where(mask, E_rnd, E_next),
                         t=jp.where(truncated, 0, t_next), key=k2)
    return new_state, E_next, reward, truncated


def geodesic_action(G: Group, E: jax.Array, alpha_max: float) -> jax.Array:
    """u(E) = -Log(E) expressed in pre-clip action units."""
    xi = G.Log(E)
    a = -xi / alpha_max
    n = jp.linalg.norm(a, axis=-1, keepdims=True)
    return a * jp.minimum(1.0, 1.0 / jp.clip(n, 1e-12))


def rollout(G: Group, policy, E0: jax.Array, params: EnvParams):
    """Deterministic rollout of a policy E -> action; returns (E_final, return)."""
    def one(carry, _):
        E, ret = carry
        a = policy(E)
        E2 = apply_action(G, E, a, params.alpha_max)
        r = -shaped_cost(G, E2, params) / params.work_scale
        return (E2, ret + r), None

    (Ef, ret), _ = jax.lax.scan(one, (E0, jp.zeros(E0.shape[0])), None,
                                length=params.max_steps)
    return Ef, ret
