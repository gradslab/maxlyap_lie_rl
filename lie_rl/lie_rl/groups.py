"""Matrix Lie group primitives for the generalized idealized task."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

PI = float(jp.pi)
WSTAR = 2.0 * PI
RINJ = PI


def _qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jp.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _qconj(q):
    return q * jp.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def small_angle_threshold(dtype) -> float:
    """Switch-over angle between the closed form and its three-term series."""
    return 4.0 * float(jp.finfo(dtype).eps) ** (1.0 / 6.0)


def _sinc(x):
    """sin(x)/x, stable at 0 (no cancellation, so any small guard works)."""
    small = jp.abs(x) < 1e-8
    xs = jp.where(small, 1.0, x)
    return jp.where(small, 1.0 - x**2 / 6.0, jp.sin(xs) / xs)


def _qexp(w):
    """Exp: R^3 -> unit quaternion, half-angle series safe at 0."""
    th = jp.linalg.norm(w, axis=-1, keepdims=True)
    half = 0.5 * th
    return jp.concatenate([w * (0.5 * _sinc(half)), jp.cos(half)], axis=-1)


def _qlog(q):
    """Log: unit quaternion -> R^3 with ||w|| <= pi (principal branch)."""
    q = jp.where(q[..., 3:4] < 0, -q, q)
    xyz = q[..., :3]
    n = jp.linalg.norm(xyz, axis=-1, keepdims=True)
    w = jp.clip(q[..., 3:4], -1.0, 1.0)
    th = 2.0 * jp.arctan2(n, w)
    small = n < 1e-8
    scale = jp.where(small, 2.0 + (2.0 / 3.0) * (n**2), th / jp.where(small, 1.0, n))
    return xyz * scale


def _qmat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jp.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )


def _qrot(q, v):
    m = _qmat(q).reshape(q.shape[:-1] + (3, 3))
    return jp.einsum("...ij,...j->...i", m, v)


def _hat(w):
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


def _V_so3(w):
    """Left Jacobian V(w) = I + ((1-cos t)/t^2) W + ((t-sin t)/t^3) W^2."""
    t = jp.linalg.norm(w, axis=-1)
    W = _hat(w)
    t2 = t * t
    small = t < small_angle_threshold(w.dtype)
    ts = jp.where(small, 1.0, t)
    c1 = jp.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0,
                  (1.0 - jp.cos(ts)) / (ts * ts))
    c2 = jp.where(small, 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0,
                  (ts - jp.sin(ts)) / (ts**3))
    eye = jp.eye(3, dtype=w.dtype)
    return eye + c1[..., None, None] * W + c2[..., None, None] * (W @ W)


def _Vinv_so3(w):
    """V(w)^{-1} = I - W/2 + (1/t^2)(1 - t sin t / (2(1-cos t))) W^2."""
    t = jp.linalg.norm(w, axis=-1)
    W = _hat(w)
    t2 = t * t
    small = t < small_angle_threshold(w.dtype)
    ts = jp.where(small, 1.0, t)
    num = ts * jp.sin(ts)
    den = 2.0 * (1.0 - jp.cos(ts))
    c = jp.where(small, 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0,
                 (1.0 - num / den) / (ts * ts))
    eye = jp.eye(3, dtype=w.dtype)
    return eye - 0.5 * W + c[..., None, None] * (W @ W)


class Group(NamedTuple):
    """A concrete instantiation of Assumption 2.1 of the theory document."""

    name: str
    m: int
    obs_dim: int
    rinj: float
    wstar: float
    compact: bool

    identity: callable
    compose: callable
    inverse: callable
    Exp: callable
    Log: callable
    obs: callable
    sample: callable
    err_parts: callable
    work_scale: callable


def _so2_identity(n):
    return jp.tile(jp.array([1.0, 0.0]), (n, 1))


def _so2_compose(a, b):
    c = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    s = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return jp.stack([c, s], axis=-1)


def _so2_inverse(a):
    return jp.stack([a[..., 0], -a[..., 1]], axis=-1)


def _so2_exp(tau):
    th = tau[..., 0]
    return jp.stack([jp.cos(th), jp.sin(th)], axis=-1)


def _so2_log(a):
    return jp.arctan2(a[..., 1], a[..., 0])[..., None]


def _so2_obs(a):
    c, s = a[..., 0], a[..., 1]
    return jp.stack([c, -s, s, c], axis=-1)


def _so2_sample(key, n, T_max):
    th = jax.random.uniform(key, (n,), minval=-PI, maxval=PI)
    return jp.stack([jp.cos(th), jp.sin(th)], axis=-1)


def _so2_err(a):
    th = jp.abs(jp.arctan2(a[..., 1], a[..., 0]))
    return th, jp.zeros_like(th)


SO2 = Group(
    name="SO2", m=1, obs_dim=4, rinj=PI, wstar=WSTAR, compact=True,
    identity=_so2_identity, compose=_so2_compose, inverse=_so2_inverse,
    Exp=_so2_exp, Log=_so2_log, obs=_so2_obs, sample=_so2_sample,
    err_parts=_so2_err, work_scale=lambda T_max: PI,
)


def _se2_rotmat(c, s):
    return jp.stack([jp.stack([c, -s], -1), jp.stack([s, c], -1)], axis=-2)


def _se2_identity(n):
    return jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), (n, 1))


def _se2_compose(a, b):
    c1, s1, t1 = a[..., 0], a[..., 1], a[..., 2:4]
    c2, s2, t2 = b[..., 0], b[..., 1], b[..., 2:4]
    c = c1 * c2 - s1 * s2
    s = c1 * s2 + s1 * c2
    R1t2 = jp.stack([c1 * t2[..., 0] - s1 * t2[..., 1],
                     s1 * t2[..., 0] + c1 * t2[..., 1]], axis=-1)
    return jp.concatenate([jp.stack([c, s], -1), t1 + R1t2], axis=-1)


def _se2_inverse(a):
    c, s, t = a[..., 0], a[..., 1], a[..., 2:4]
    ti = jp.stack([-(c * t[..., 0] + s * t[..., 1]),
                   -(-s * t[..., 0] + c * t[..., 1])], axis=-1)
    return jp.concatenate([jp.stack([c, -s], -1), ti], axis=-1)


def _se2_exp(tau):
    th, v = tau[..., 0], tau[..., 1:3]
    a = _sinc(0.5 * th)
    ch, sh = jp.cos(0.5 * th), jp.sin(0.5 * th)
    t = a[..., None] * jp.stack([ch * v[..., 0] - sh * v[..., 1],
                                 sh * v[..., 0] + ch * v[..., 1]], axis=-1)
    return jp.concatenate([jp.stack([jp.cos(th), jp.sin(th)], -1), t], axis=-1)


def _se2_log(a):
    th = jp.arctan2(a[..., 1], a[..., 0])
    t = a[..., 2:4]
    inv = 1.0 / _sinc(0.5 * th)
    ch, sh = jp.cos(0.5 * th), jp.sin(0.5 * th)
    v = inv[..., None] * jp.stack([ch * t[..., 0] + sh * t[..., 1],
                                   -sh * t[..., 0] + ch * t[..., 1]], axis=-1)
    return jp.concatenate([th[..., None], v], axis=-1)


def _se2_obs(a):
    c, s = a[..., 0], a[..., 1]
    return jp.concatenate([jp.stack([c, -s, s, c], -1), a[..., 2:4]], axis=-1)


def _se2_sample(key, n, T_max):
    k1, k2, k3 = jax.random.split(key, 3)
    th = jax.random.uniform(k1, (n,), minval=-PI, maxval=PI)
    d = jax.random.normal(k2, (n, 2))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    r = T_max * jax.random.uniform(k3, (n, 1)) ** (1.0 / 2.0)
    return jp.concatenate([jp.stack([jp.cos(th), jp.sin(th)], -1), d * r], axis=-1)


def _se2_err(a):
    return jp.abs(jp.arctan2(a[..., 1], a[..., 0])), jp.linalg.norm(a[..., 2:4], axis=-1)


SE2 = Group(
    name="SE2", m=3, obs_dim=6, rinj=PI, wstar=WSTAR, compact=False,
    identity=_se2_identity, compose=_se2_compose, inverse=_se2_inverse,
    Exp=_se2_exp, Log=_se2_log, obs=_se2_obs, sample=_se2_sample,
    err_parts=_se2_err,
    work_scale=lambda T_max: float((PI**2 + (0.5 * PI * T_max) ** 2) ** 0.5),
)


def _se3_identity(n):
    return jp.tile(jp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (n, 1))


def _se3_compose(a, b):
    q1, t1 = a[..., :4], a[..., 4:7]
    q2, t2 = b[..., :4], b[..., 4:7]
    return jp.concatenate([_qmul(q1, q2), t1 + _qrot(q1, t2)], axis=-1)


def _se3_inverse(a):
    q, t = a[..., :4], a[..., 4:7]
    qi = _qconj(q)
    return jp.concatenate([qi, -_qrot(qi, t)], axis=-1)


def _se3_exp(tau):
    w, v = tau[..., :3], tau[..., 3:6]
    q = _qexp(w)
    t = jp.einsum("...ij,...j->...i", _V_so3(w), v)
    return jp.concatenate([q, t], axis=-1)


def _se3_log(a):
    q, t = a[..., :4], a[..., 4:7]
    w = _qlog(q)
    v = jp.einsum("...ij,...j->...i", _Vinv_so3(w), t)
    return jp.concatenate([w, v], axis=-1)


def _se3_obs(a):
    return jp.concatenate([_qmat(a[..., :4]), a[..., 4:7]], axis=-1)


def _se3_sample(key, n, T_max):
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (n, 4))
    q = q / jp.linalg.norm(q, axis=-1, keepdims=True)
    d = jax.random.normal(k2, (n, 3))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    r = T_max * jax.random.uniform(k3, (n, 1)) ** (1.0 / 3.0)
    return jp.concatenate([q, d * r], axis=-1)


def _se3_err(a):
    q = jp.where(a[..., 3:4] < 0, -a[..., :4], a[..., :4])
    th = 2.0 * jp.arctan2(jp.linalg.norm(q[..., :3], axis=-1), jp.clip(q[..., 3], -1.0, 1.0))
    return th, jp.linalg.norm(a[..., 4:7], axis=-1)


SE3 = Group(
    name="SE3", m=6, obs_dim=12, rinj=PI, wstar=WSTAR, compact=False,
    identity=_se3_identity, compose=_se3_compose, inverse=_se3_inverse,
    Exp=_se3_exp, Log=_se3_log, obs=_se3_obs, sample=_se3_sample,
    err_parts=_se3_err,
    work_scale=lambda T_max: float((PI**2 + (0.5 * PI * T_max) ** 2) ** 0.5),
)


def _so3_identity(n):
    return jp.tile(jp.array([0.0, 0.0, 0.0, 1.0]), (n, 1))


def _so3_sample(key, n, T_max):
    q = jax.random.normal(key, (n, 4))
    return q / jp.linalg.norm(q, axis=-1, keepdims=True)


def _so3_err(q):
    qc = jp.where(q[..., 3:4] < 0, -q, q)
    th = 2.0 * jp.arctan2(jp.linalg.norm(qc[..., :3], axis=-1),
                          jp.clip(qc[..., 3], -1.0, 1.0))
    return th, jp.zeros_like(th)


SO3 = Group(
    name="SO3", m=3, obs_dim=9, rinj=PI, wstar=WSTAR, compact=True,
    identity=_so3_identity, compose=_qmul, inverse=_qconj,
    Exp=_qexp, Log=_qlog, obs=_qmat, sample=_so3_sample,
    err_parts=_so3_err, work_scale=lambda T_max: PI,
)


GROUPS = {"SO2": SO2, "SO3": SO3, "SE2": SE2, "SE3": SE3}
