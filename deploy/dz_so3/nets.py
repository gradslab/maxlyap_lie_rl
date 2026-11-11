"""Tiny pure-JAX MLPs as pytrees (no flax dependency in the hot path)."""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def init_mlp(key, sizes, scale="orthogonal", out_gain=0.01):
    """List of (W, b). Orthogonal init (PPO standard) or lecun."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        k = keys[i]
        if scale == "orthogonal":
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            w = jax.nn.initializers.orthogonal(gain)(k, (din, dout))
        else:
            w = jax.random.normal(k, (din, dout)) / np.sqrt(din)
        params.append((w, jp.zeros(dout)))
    return params


def mlp_apply(params, x, act, out_act=None):
    h = x
    for w, b in params[:-1]:
        h = act(h @ w + b)
    w, b = params[-1]
    h = h @ w + b
    if out_act is not None:
        h = out_act(h)
    return h


def spectral_norms(params):
    """Largest singular value per layer weight."""
    return [float(jp.linalg.norm(w, ord=2)) for w, _ in params]


def lipschitz_product(params):
    """Global Lipschitz upper bound for 1-Lipschitz activations."""
    out = 1.0
    for s in spectral_norms(params):
        out *= s
    return out
