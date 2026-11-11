"""BlueROV2 Heavy — hydrodynamics, thrust allocation, and attitude inner loop."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).with_name("bluerov2_heavy.xml")

G = 9.82
RHO = 1000.0
MASS = 13.5
VOLUME = 0.0134
BUOYANCY = RHO * G * VOLUME
Z_B = 0.01

D_LIN = np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0])
D_QUAD = np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5])
ADDED_MASS_LIN = np.array([6.36, 7.12, 18.68])

THRUST_MAX = 30.0
N_THRUSTERS = 8


def load(z_b: float = Z_B):
    """Load the model. `z_b` is the CoB height above CoM, in metres."""
    xml = XML.read_text().replace('name="cob" pos="0 0 0.01"',
                                  f'name="cob" pos="0 0 {z_b}"')
    m = mujoco.MjModel.from_xml_string(xml, {})
    return m, mujoco.MjData(m)


def allocation_matrix(m) -> np.ndarray:
    """6x8 body-frame wrench per unit thrust, read from the model's actuators."""
    B = np.zeros((6, N_THRUSTERS))
    for i in range(m.nu):
        sid = m.actuator_trnid[i, 0]
        r = m.site_pos[sid]
        e = m.actuator_gear[i, :3]
        B[:3, i] = e
        B[3:, i] = np.cross(r, e)
    return B


def apply_hydro(m, d, prev_vel_body: np.ndarray | None = None) -> np.ndarray:
    """Apply buoyancy + Fossen damping + translational added mass for one step."""
    R = d.xmat[1].reshape(3, 3)
    v_w = d.cvel[1, 3:].copy()
    w_w = d.cvel[1, :3].copy()
    v_b = R.T @ v_w
    w_b = R.T @ w_w
    nu = np.concatenate([v_b, w_b])

    damp = -(D_LIN * nu + D_QUAD * nu * np.abs(nu))
    f_b, t_b = damp[:3], damp[3:]


    r_cob_w = R @ np.array([0.0, 0.0, m.site_pos[0, 2]])
    f_buoy_w = np.array([0.0, 0.0, BUOYANCY])
    t_buoy_w = np.cross(r_cob_w, f_buoy_w)

    d.xfrc_applied[1, :3] = R @ f_b + f_buoy_w
    d.xfrc_applied[1, 3:] = R @ t_b + t_buoy_w
    return v_b


def allocate(B: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    """Least-squares thrust allocation, saturated at +-THRUST_MAX."""
    u = np.linalg.pinv(B) @ wrench
    return np.clip(u, -THRUST_MAX, THRUST_MAX)


class AttitudeInnerLoop:
    """Classical PD attitude tracker — the inner loop our RL policy sits above."""

    def __init__(self, m, kp=(600.0, 600.0, 600.0), kd=(80.0, 80.0, 80.0),
                 station_keep: bool = False, kp_pos=120.0, kd_pos=90.0):
        self.B = allocation_matrix(m)
        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        self.station_keep = station_keep
        self.kp_pos, self.kd_pos = float(kp_pos), float(kd_pos)
        self.p_des = None

    def __call__(self, d, R_des: np.ndarray) -> np.ndarray:
        R = d.xmat[1].reshape(3, 3)
        Re = R.T @ R_des
        ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        if ang < 1e-9:
            e = np.zeros(3)
        else:
            axis = np.array([Re[2, 1] - Re[1, 2],
                             Re[0, 2] - Re[2, 0],
                             Re[1, 0] - Re[0, 1]]) / (2 * np.sin(ang))
            e = axis * ang
        w_b = R.T @ d.cvel[1, :3]
        torque = self.kp * e - self.kd * w_b
        if self.station_keep:
            if self.p_des is None:
                self.p_des = d.qpos[:3].copy()
            v_w = d.cvel[1, 3:]
            f_w = self.kp_pos * (self.p_des - d.qpos[:3]) - self.kd_pos * v_w
            force = R.T @ f_w
        else:
            force = np.zeros(3)
        wrench = np.concatenate([force, torque])
        return allocate(self.B, wrench)


def apply_added_mass(m, d) -> None:
    """Translational added mass and its Coriolis terms, exactly, for the step about to be taken."""
    mujoco.mj_forward(m, d)
    R = d.xmat[1].reshape(3, 3)
    v_b = R.T @ d.qvel[:3]
    w_b = d.qvel[3:6].copy()
    a0_b = R.T @ d.qacc[:3] - np.cross(w_b, v_b)
    MA = ADDED_MASS_LIN
    nudot = np.linalg.solve(MASS * np.eye(3) + np.diag(MA), MASS * a0_b - np.cross(w_b, MA * v_b))
    f_b = MASS * (nudot - a0_b)
    t_b = -np.cross(v_b, MA * v_b)
    d.xfrc_applied[1, :3] += R @ f_b
    d.xfrc_applied[1, 3:] += R @ t_b


class PoseInnerLoop:
    """PD pose tracker for the SE(3) policy: the attitude law of AttitudeInnerLoop plus a translational"""

    def __init__(self, m, kp=600.0, kd=80.0, kp_pos=400.0, kd_pos=160.0):
        self.B = allocation_matrix(m)
        self.kp, self.kd, self.kp_pos, self.kd_pos = float(kp), float(kd), float(kp_pos), float(kd_pos)
        self.sat = 0; self.calls = 0

    def __call__(self, d, R_des: np.ndarray, p_des: np.ndarray) -> np.ndarray:
        R = d.xmat[1].reshape(3, 3)
        Re = R.T @ R_des
        ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        if ang < 1e-9:
            e = np.zeros(3)
        else:
            e = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * np.sin(ang)) * ang
        w_b = d.qvel[3:6]
        torque = self.kp * e - self.kd * w_b
        v_b = R.T @ d.qvel[:3]
        ff_w = np.array([0.0, 0.0, MASS * G - BUOYANCY])
        force = self.kp_pos * (R.T @ (p_des - d.qpos[:3])) - self.kd_pos * v_b + R.T @ ff_w
        u_raw = np.linalg.pinv(self.B) @ np.concatenate([force, torque])
        self.calls += 1; self.sat += int(np.any(np.abs(u_raw) > THRUST_MAX))
        return np.clip(u_raw, -THRUST_MAX, THRUST_MAX)


def added_mass_self_test() -> None:
    """A constant body force from rest, damping and buoyancy off: the acceleration must be F/(m + M_A)"""
    for axis in range(3):
        m, d = load(); m.opt.gravity[:] = 0
        F = np.zeros(3); F[axis] = 10.0
        for _ in range(200):
            d.xfrc_applied[1, :] = 0; d.xfrc_applied[1, :3] = d.xmat[1].reshape(3, 3) @ F
            apply_added_mass(m, d); mujoco.mj_step(m, d)
        t = 200 * m.opt.timestep
        a = 2 * d.qpos[axis] / t ** 2
        want = 10.0 / (MASS + ADDED_MASS_LIN[axis])
        print(f"  axis {axis}: acceleration {a:.5f} m/s^2, expected {want:.5f}  ({100*abs(a-want)/want:.2f}% off)")
        assert abs(a - want) / want < 0.01


def self_test() -> None:
    """Reproduce the paper's Table 4 maxima from the model geometry."""
    m, _ = load()
    B = allocation_matrix(m)
    got = np.abs(B).sum(axis=1) * THRUST_MAX
    paper = np.array([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])
    names = ["surge X", "sway Y", "heave Z", "roll K", "pitch M", "yaw N"]
    print(f"{'':10s}{'model':>9s}{'paper':>8s}{'err':>8s}")
    for n, g_, p in zip(names, got, paper):
        print(f"  {n:8s}{g_:9.2f}{p:8.1f}{abs(g_ - p) / p * 100:7.1f}%")
    assert np.all(np.abs(got - paper) / paper < 0.04), "geometry does not match Table 4"
    assert np.linalg.matrix_rank(B) == 6, "not fully actuated"
    print(f"\nrank {np.linalg.matrix_rank(B)} (fully actuated) | "
          f"buoyancy {BUOYANCY:.2f} N vs weight {MASS * G:.2f} N "
          f"(net {MASS * G - BUOYANCY:+.2f} N) | all maxima within 4%  OK")


if __name__ == "__main__":
    self_test()
    added_mass_self_test()
