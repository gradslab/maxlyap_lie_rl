"""SO(3) geometry primitives (JAX, float64)."""

from __future__ import annotations

import jax
import jax.numpy as jp


def hat(w: jax.Array) -> jax.Array:
    """R^3 -> so(3)."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    z = jp.zeros_like(wx)
    return jp.stack(
        [
            jp.stack([z, -wz, wy], axis=-1),
            jp.stack([wz, z, -wx], axis=-1),
            jp.stack([-wy, wx, z], axis=-1),
        ],
        axis=-2,
    )


def vee(W: jax.Array) -> jax.Array:
    """so(3) -> R^3."""
    return jp.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jp.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_conj(q: jax.Array) -> jax.Array:
    return q * jp.array([-1.0, -1.0, -1.0, 1.0])


def quat_canonical(q: jax.Array) -> jax.Array:
    """Force scalar part >= 0 (principal hemisphere)."""
    return jp.where(q[..., 3:4] < 0, -q, q)


def exp_quat(v: jax.Array) -> jax.Array:
    """Rotation-vector -> unit quaternion, Exp(hat(v))."""
    t = jp.sum(v * v, axis=-1, keepdims=True)
    small = t < 1e-12
    t_safe = jp.where(small, 1.0, t)
    th = jp.sqrt(t_safe)
    k = jp.where(small, 0.5 - t / 48.0 + t * t / 3840.0, jp.sin(0.5 * th) / th)
    w = jp.where(small, 1.0 - t / 8.0 + t * t / 384.0, jp.cos(0.5 * th))
    return jp.concatenate([v * k, w], axis=-1)


def log_quat(q: jax.Array) -> jax.Array:
    """Unit quaternion -> rotation vector (principal branch, ||.|| <= pi)."""
    q = quat_canonical(q)
    xyz = q[..., :3]
    w = jp.clip(q[..., 3:4], 1e-12)
    s2 = jp.sum(xyz * xyz, axis=-1, keepdims=True)
    small = s2 < 1e-12
    s2_safe = jp.where(small, 1.0, s2)
    s = jp.sqrt(s2_safe)
    scale_large = 2.0 * jp.arctan2(s, w) / s
    scale_small = (2.0 / w) * (1.0 - s2 / (3.0 * w * w))
    return xyz * jp.where(small, scale_small, scale_large)


def quat_to_mat(q: jax.Array) -> jax.Array:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = jp.stack(
        [
            jp.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            jp.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            jp.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )
    return m


def angle_of(q: jax.Array) -> jax.Array:
    """Geodesic angle of the rotation q, in [0, pi]."""
    q = quat_canonical(q)
    return 2.0 * jp.arctan2(jp.linalg.norm(q[..., :3], axis=-1), q[..., 3])


def d_geo(q1: jax.Array, q2: jax.Array) -> jax.Array:
    return angle_of(quat_mul(quat_conj(q1), q2))


def fro_sq(q: jax.Array) -> jax.Array:
    """||E - I||_F^2 = 8 sin^2(theta / 2) = 8 (1 - w^2) for the unit quaternion."""
    w = quat_canonical(q)[..., 3]
    return 8.0 * jp.clip(1.0 - w * w, 0.0)


def ell(q: jax.Array) -> jax.Array:
    """Running cost ell(s) = ||E - I||_F (E1: no Euclidean part)."""
    return jp.sqrt(jp.clip(fro_sq(q), 1e-30))


def sample_ball(key: jax.Array, n: int, radius: float, r_min: float = 0.0) -> jax.Array:
    """Uniform sample in the annulus r_min <= ||z|| <= radius of R^3."""
    k1, k2 = jax.random.split(key)
    d = jax.random.normal(k1, (n, 3))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    u = jax.random.uniform(k2, (n, 1))
    r = (r_min**3 + u * (radius**3 - r_min**3)) ** (1.0 / 3.0)
    return d * r


def sample_quat_uniform(key: jax.Array, n: int) -> jax.Array:
    """Uniform (Haar) random quaternions: normalized 4-D Gaussians (P2 reset)."""
    q = jax.random.normal(key, (n, 4))
    return q / jp.linalg.norm(q, axis=-1, keepdims=True)
