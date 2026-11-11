"""Closed loops of a trained policy, without and with an actuator gain error or misalignment."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import jax.numpy as jp
import numpy as np

from . import env as E
from .ppo import actor_mean, critic_value

RES = Path(__file__).parents[1] / "results"


def parse_variant(v: str) -> dict:
    if v == "base":
        return dict(gain=1.0, mis_deg=0.0)
    mg = re.fullmatch(r"g([0-9.]+)", v)
    if mg:
        return dict(gain=float(mg.group(1)), mis_deg=0.0)
    ma = re.fullmatch(r"a([0-9]+)", v)
    if ma:
        return dict(gain=1.0, mis_deg=0.0, axis_seed=int(ma.group(1)))
    mm = re.fullmatch(r"m([0-9.]+)", v)
    if mm:
        return dict(gain=1.0, mis_deg=float(mm.group(1)))
    raise ValueError(f"unknown variant {v!r}")


def _axis_rotation(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    n = np.ones(3) / np.sqrt(3.0)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mis_matrix(group: str, deg: float) -> np.ndarray:
    """Orthogonal m x m matrix acting on the increment (coordinates as in groups.py)."""
    if deg == 0.0:
        return None
    th = np.deg2rad(deg)
    if group == "SO3":
        return _axis_rotation(deg)
    if group == "SE2":
        M = np.eye(3)
        M[1:, 1:] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        return M
    if group == "SE3":
        R = _axis_rotation(deg)
        M = np.zeros((6, 6)); M[:3, :3] = R; M[3:, 3:] = R
        return M
    raise ValueError(f"misalignment is undefined on {group}")


def load_agent(d: str, seed: int):
    return pickle.load(open(RES / d / f"agent_seed{seed}.pkl", "rb"))


def actor_of(agent):
    return [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in agent.actor]


class ClosedLoop:
    """The corrected policy u(E) = mu(vec E) - mu(vec Id) and its closed loop under a variant."""

    def __init__(self, group: str, d: str, seed: int, variant: str = "base"):
        self.group, self.d, self.seed, self.variant = group, d, seed, variant
        self.ep = E.EnvParams(group=group)
        self.G = self.ep.G
        self.m = self.G.m
        self.amax = self.ep.alpha_max
        self.agent = load_agent(d, seed)
        self.af = actor_of(self.agent)
        self.bias = actor_mean(self.af, E.obs(self.G, self.G.identity(1)))[0]
        spec = parse_variant(variant)
        self.gain = spec["gain"]
        if spec.get("axis_seed") is not None:
            self.gain = jp.asarray(np.random.default_rng(1000 + spec["axis_seed"]).uniform(0.7, 1.3, self.m))
        M = mis_matrix(group, spec["mis_deg"])
        self.M = None if M is None else jp.asarray(M)

    def action(self, X):
        return actor_mean(self.af, E.obs(self.G, X)) - self.bias

    def step(self, X):
        G, amax = self.G, self.amax
        a = self.action(X)
        if self.variant == "base":
            return E.apply_action(G, X, a, amax)
        tau = E.clip_tau(G, a, amax)
        if self.M is not None:
            tau = tau @ self.M.T
        tau = self.gain * tau
        n = jp.linalg.norm(tau, axis=-1, keepdims=True)
        tau = tau * jp.minimum(1.0, amax / jp.clip(n, 1e-12))
        return G.compose(X, G.Exp(tau))

    def chart_map(self):
        G = self.G
        return lambda z: G.Log(self.step(G.Exp(z)))

    def has_value_critic(self) -> bool:
        """PPO and TRPO store a state-value critic V(obs); SAC and TD3 store Q(obs, a)."""
        c = self.agent.critic
        return int(np.asarray(c[0][0]).shape[0]) == int(self.G.obs_dim)

    def value_candidate(self):
        """V_v(z) = c(Id) - c(Exp z), from the critic trained with the policy (discounted return of the"""
        cp = [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in self.agent.critic]
        G = self.G
        c0 = critic_value(cp, E.obs(G, G.identity(1)))[0]
        return lambda z: c0 - critic_value(cp, E.obs(G, G.Exp(z)))
