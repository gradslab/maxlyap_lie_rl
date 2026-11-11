"""MuJoCo <-> dz_so3 adapter. THE ONLY PLACE CONVENTIONS ARE TRANSLATED."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jp
from dz_so3 import so3 as S

QPOS_QUAT = slice(3, 7)
QVEL_ANG = slice(3, 6)


def mj_quat_to_dz(q_mj):
    """MuJoCo [w,x,y,z] -> dz_so3 [x,y,z,w]."""
    q = np.asarray(q_mj, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def dz_quat_to_mj(q_dz):
    """dz_so3 [x,y,z,w] -> MuJoCo [w,x,y,z]."""
    q = np.asarray(q_dz, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def attitude_dz(data):
    """Current attitude as a dz_so3 quaternion [x,y,z,w]."""
    return mj_quat_to_dz(np.array(data.qpos[QPOS_QUAT]))


def set_attitude_dz(data, q_dz):
    """Write a dz_so3 quaternion into MuJoCo, normalised."""
    q = np.asarray(q_dz, dtype=float)
    q = q / np.linalg.norm(q)
    data.qpos[QPOS_QUAT] = dz_quat_to_mj(q)


def error_vector(q_dz, q_goal_dz):
    """z = Log(E) with E = R_goal^T R -- THE SAME error convention the certificate uses."""
    qg = jp.asarray(np.asarray(q_goal_dz, dtype=float))
    q = jp.asarray(np.asarray(q_dz, dtype=float))
    qE = S.quat_mul(S.quat_conj(qg), q)
    return np.asarray(S.log_quat(qE), dtype=float)


def omega_body(data):
    """Free-joint angular velocity as MuJoCo stores it."""
    return np.array(data.qvel[QVEL_ANG], dtype=float)


def inertia_diag(model):
    """Body diagonal inertia (body 1 = the spacecraft; body 0 is the world)."""
    return np.array(model.body_inertia[1], dtype=float)


def angular_momentum_body(model, data):
    """H_body = J omega, in whatever frame omega is reported (see omega_body)."""
    return inertia_diag(model) * omega_body(data)


def kinetic_energy(model, data):
    """T = 1/2 omega^T J omega."""
    w = omega_body(data)
    return float(0.5 * np.dot(w, inertia_diag(model) * w))


def np_quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def np_quat_conj(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def np_exp_quat(v):
    t = np.sum(v * v, axis=-1, keepdims=True)
    small = t < 1e-12
    th = np.sqrt(np.where(small, 1.0, t))
    k = np.where(small, 0.5 - t / 48.0 + t * t / 3840.0, np.sin(0.5 * th) / th)
    w = np.where(small, 1.0 - t / 8.0 + t * t / 384.0, np.cos(0.5 * th))
    return np.concatenate([v * k, w], axis=-1)


def np_log_quat(q):
    q = np.where(q[..., 3:4] < 0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3:4], 1e-12, None)
    s2 = np.sum(xyz * xyz, axis=-1, keepdims=True)
    small = s2 < 1e-12
    s = np.sqrt(np.where(small, 1.0, s2))
    scale = np.where(small, (2.0 / w) * (1.0 - s2 / (3.0 * w * w)),
                     2.0 * np.arctan2(s, w) / s)
    return xyz * scale


def np_error_vector(q, q_goal):
    """z = Log(R_goal^T R), NumPy mirror of error_vector()."""
    return np_log_quat(np_quat_mul(np_quat_conj(np.asarray(q_goal, dtype=float)),
                                   np.asarray(q, dtype=float)))
