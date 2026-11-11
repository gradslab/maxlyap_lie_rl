"""Inner-loop attitude controller: execute one commanded increment within dt_outer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import mujoco
import jax.numpy as jp
from dz_so3 import so3 as S
import adapter as A

ALPHA_MAX = float(np.pi * 0.1)


def slew_time(J, tau_max, angle=ALPHA_MAX):
    """Rest-to-rest bang-bang slew time per axis: t = 2 sqrt(angle * J / tau_max)."""
    return 2.0 * np.sqrt(angle * np.asarray(J) / tau_max)


def derive_gains(J, tau_max, angle=ALPHA_MAX, zeta=1.0):
    """Kp, Kd from the torque authority. Returns (Kp, Kd, omega_n, t_slew)."""
    t_slew = slew_time(J, tau_max, angle)
    omega_n = 2.0 / t_slew
    Kp = np.asarray(J) * omega_n ** 2
    Kd = 2.0 * zeta * np.asarray(J) * omega_n
    return Kp, Kd, omega_n, t_slew


def torque_limit(model):
    """Per-actuator ctrlrange magnitude (symmetric by construction in model.xml)."""
    return float(np.abs(model.actuator_ctrlrange[0]).max())


def step_outer(model, data, tau_cmd, dt_outer, Kp, Kd, tau_max):
    """Hold `tau_cmd` for one outer step; return diagnostics."""
    J = A.inertia_diag(model)
    q0 = A.attitude_dz(data)
    q_des = np.asarray(S.quat_mul(jp.asarray(q0),
                                  jp.asarray(np.asarray(S.exp_quat(jp.asarray(tau_cmd))))),
                       dtype=float)
    n = int(round(dt_outer / model.opt.timestep))
    for _ in range(n):
        q = A.attitude_dz(data)
        e_R = A.np_error_vector(q, q_des)
        w = A.omega_body(data)
        tau = -Kp * e_R - Kd * w + np.cross(w, J * w)
        data.ctrl[:] = np.clip(tau, -tau_max, tau_max)
        mujoco.mj_step(model, data)
    q1 = A.attitude_dz(data)
    achieved = A.np_error_vector(q1, q0)
    cmd_norm = float(np.linalg.norm(tau_cmd))
    return dict(q1=q1,
                achieved=achieved,
                achieved_norm=float(np.linalg.norm(achieved)),
                commanded_norm=cmd_norm,
                tracking_fraction=(float(np.linalg.norm(achieved)) / cmd_norm) if cmd_norm > 1e-12 else np.nan,
                residual_rate_norm=float(np.linalg.norm(A.omega_body(data))),
                terminal_err_to_des=float(np.linalg.norm(A.np_error_vector(q1, q_des))))
