"""THE GATE. Validate the MuJoCo plant and the adapter BEFORE any nu number may be produced.

Usage: python spacecraft/sc_validate.py
Writes spacecraft/results/sc_validate.json  (exit 0 only if every check passes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mujoco
import jax.numpy as jp
from dz_so3 import so3 as S
import adapter as A

XML = HERE / "model.xml"
rng = np.random.default_rng(0)


def fresh():
    m = mujoco.MjModel.from_xml_path(str(XML))
    return m, mujoco.MjData(m)


def rand_quat(n):
    z = rng.normal(size=(n, 3))
    z = z / np.linalg.norm(z, axis=1, keepdims=True) * rng.uniform(0.05, 3.0, (n, 1))
    return np.asarray(S.exp_quat(jp.asarray(z)), dtype=float)


def t1_quaternion_convention():
    q = rand_quat(2000)
    back = A.mj_quat_to_dz(A.dz_quat_to_mj(q))
    err_roundtrip = float(np.abs(back - q).max())
    worst = 0.0
    for i in range(200):
        Rd = np.asarray(S.quat_to_mat(jp.asarray(q[i])), dtype=float)
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, A.dz_quat_to_mj(q[i]))
        worst = max(worst, float(np.abs(Rd - Rm.reshape(3, 3)).max()))
    return dict(roundtrip_max_abs=err_roundtrip, rotmat_max_abs=worst,
                passed=bool(err_roundtrip < 1e-12 and worst < 1e-10))


def t2_omega_frame():
    """Drive a known omega about a principal axis from a NON-identity attitude and see whether the"""
    m, d = fresh()
    q0 = rand_quat(1)[0]
    A.set_attitude_dz(d, q0)
    w = np.array([0.0, 0.0, 0.31])
    d.qvel[A.QVEL_ANG] = w
    mujoco.mj_forward(m, d)
    n, dt = 50, m.opt.timestep
    for _ in range(n):
        mujoco.mj_step(m, d)
    q1 = A.attitude_dz(d)
    dq = np.asarray(S.exp_quat(jp.asarray(w * n * dt)), dtype=float)
    q_body = np.asarray(S.quat_mul(jp.asarray(q0), jp.asarray(dq)), dtype=float)
    q_world = np.asarray(S.quat_mul(jp.asarray(dq), jp.asarray(q0)), dtype=float)
    e_body = float(np.linalg.norm(A.error_vector(q1, q_body)))
    e_world = float(np.linalg.norm(A.error_vector(q1, q_world)))
    frame = "body" if e_body < e_world else "world"
    return dict(err_if_body=e_body, err_if_world=e_world, frame_determined=frame,
                margin_ratio=float(max(e_body, e_world) / max(min(e_body, e_world), 1e-18)),
                passed=bool(min(e_body, e_world) < 1e-6 and
                            max(e_body, e_world) > 100 * min(e_body, e_world)))


def t3_zero_torque_agreement():
    """MuJoCo's attitude increment must equal dz_so3's Exp(omega dt) with tau = 0."""
    out = []
    for axis in range(3):
        m, d = fresh()
        q0 = rand_quat(1)[0]
        A.set_attitude_dz(d, q0)
        w = np.zeros(3); w[axis] = 0.25
        d.qvel[A.QVEL_ANG] = w
        mujoco.mj_forward(m, d)
        n, dt = 100, m.opt.timestep
        for _ in range(n):
            mujoco.mj_step(m, d)
        q1 = A.attitude_dz(d)
        dq = np.asarray(S.exp_quat(jp.asarray(w * n * dt)), dtype=float)
        q_pred = np.asarray(S.quat_mul(jp.asarray(q0), jp.asarray(dq)), dtype=float)
        out.append(float(np.linalg.norm(A.error_vector(q1, q_pred))))
    return dict(per_axis_geodesic_err=out, max_err=max(out), passed=bool(max(out) < 1e-9))


def t4_invariants():
    m, d = fresh()
    A.set_attitude_dz(d, rand_quat(1)[0])
    d.qvel[A.QVEL_ANG] = np.array([0.20, -0.13, 0.09])
    mujoco.mj_forward(m, d)
    H0, T0 = A.angular_momentum_body(m, d), A.kinetic_energy(m, d)
    n0, hs, ts, qn = np.linalg.norm(H0), [], [], []
    for k in range(20000):
        mujoco.mj_step(m, d)
        if k % 200 == 0:
            hs.append(abs(np.linalg.norm(A.angular_momentum_body(m, d)) - n0) / n0)
            ts.append(abs(A.kinetic_energy(m, d) - T0) / T0)
            qn.append(abs(np.linalg.norm(d.qpos[A.QPOS_QUAT]) - 1.0))
    return dict(steps=20000, sim_time_s=20000 * m.opt.timestep,
                rel_drift_angular_momentum=float(max(hs)),
                rel_drift_kinetic_energy=float(max(ts)),
                quat_norm_drift=float(max(qn)),
                passed=bool(max(hs) < 1e-8 and max(ts) < 1e-8 and max(qn) < 1e-10))


def t5_closed_form():
    """Axisymmetric torque-free precession: transverse omega rotates at lambda = (J3-J1)/J1 * w3."""
    xml = (XML.read_text()
           .replace('diaginertia="0.0110 0.0140 0.0075"', 'diaginertia="0.0110 0.0110 0.0075"'))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    J = A.inertia_diag(m)
    w3, wt = 0.30, 0.12
    d.qvel[A.QVEL_ANG] = np.array([wt, 0.0, w3])
    mujoco.mj_forward(m, d)
    lam = (J[2] - J[0]) / J[0] * w3
    T = 2.0
    n = int(T / m.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(m, d)
    w = A.omega_body(d)
    ang = float(np.arctan2(w[1], w[0]))
    pred = float(np.arctan2(np.sin(lam * T), np.cos(lam * T)))
    err = abs(np.arctan2(np.sin(ang - pred), np.cos(ang - pred)))
    mag = abs(np.hypot(w[0], w[1]) - wt) / wt
    return dict(lambda_analytic=float(lam), phase_err_rad=err, transverse_mag_rel_err=float(mag),
                passed=bool(err < 1e-4 and mag < 1e-6))


def t6_numpy_fast_path():
    """The NumPy mirrors used inside the inner loop must equal dz_so3's JAX originals."""
    z = rng.normal(size=(4000, 3))
    z = z / np.linalg.norm(z, axis=1, keepdims=True) * rng.uniform(1e-7, 3.0, (4000, 1))
    qa, qb = rand_quat(4000), rand_quat(4000)
    e = {}
    e["exp_quat"] = float(np.abs(np.asarray(S.exp_quat(jp.asarray(z))) - A.np_exp_quat(z)).max())
    e["log_quat"] = float(np.abs(np.asarray(S.log_quat(jp.asarray(qa))) - A.np_log_quat(qa)).max())
    e["quat_mul"] = float(np.abs(np.asarray(S.quat_mul(jp.asarray(qa), jp.asarray(qb)))
                                 - A.np_quat_mul(qa, qb)).max())
    e["quat_conj"] = float(np.abs(np.asarray(S.quat_conj(jp.asarray(qa))) - A.np_quat_conj(qa)).max())
    ev = np.stack([A.error_vector(qa[i], qb[i]) for i in range(300)])
    e["error_vector"] = float(np.abs(ev - A.np_error_vector(qa[:300], qb[:300])).max())
    return dict(max_abs_diff=e, worst=max(e.values()), passed=bool(max(e.values()) < 1e-12))


def main():
    res = {"T1_quaternion_convention": t1_quaternion_convention(),
           "T2_omega_frame": t2_omega_frame(),
           "T3_zero_torque_agreement": t3_zero_torque_agreement(),
           "T4_invariants": t4_invariants(),
           "T5_closed_form": t5_closed_form(),
           "T6_numpy_fast_path": t6_numpy_fast_path()}
    allp = all(v["passed"] for v in res.values())
    res["ALL_PASSED"] = allp
    import re as _re
    _x = (HERE / "model.xml").read_text()
    _J = [float(v) for v in _re.search(r'diaginertia="([^"]+)"', _x).group(1).split()]
    _tau = abs(float(_re.search(r'ctrlrange="(-?[\d.]+)', _x).group(1)))
    res["inertia"] = _J
    res["tau_max"] = _tau
    res["branch"] = "SC-1 (isolated); empirical out-of-model study, NOT a verification claim"
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {k}")
            for kk, vv in v.items():
                if kk != "passed":
                    print(f"        {kk}: {vv}")
    print(f"\n  ALL_PASSED = {allp}")
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "results/sc_validate.json", "w"), indent=2)
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()
