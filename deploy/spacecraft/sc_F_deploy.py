"""CubeSat deployment under conditions the kinematic model does not assume.

Usage: python spacecraft/sc_F_deploy.py OUT_JSON [--seed 900] [--episodes 200] [--steps 40] [--dt-outer 15.7]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import mujoco
import jax.numpy as jp
from dz_so3 import so3 as S
import adapter as A
import inner_loop as IL
import sc_deploy_study_final as SD
from sc_nu_measure_l4dc import load_policy, commanded_tau, Q_GOAL

EXEC_CMD_FLOOR = SD.EXEC_CMD_FLOOR


def step_outer_track(model, data, tau_cmd, dt_outer, Kp, Kd, tau_max):
    """inner_loop.step_outer with the same control law, also returning the largest body rate of the period."""
    J = A.inertia_diag(model); q0 = A.attitude_dz(data)
    q_des = np.asarray(S.quat_mul(jp.asarray(q0), jp.asarray(np.asarray(S.exp_quat(jp.asarray(tau_cmd))))), dtype=float)
    wmax = 0.0
    for _ in range(int(round(dt_outer / model.opt.timestep))):
        q = A.attitude_dz(data); e_R = A.np_error_vector(q, q_des); w = A.omega_body(data)
        data.ctrl[:] = np.clip(-Kp * e_R - Kd * w + np.cross(w, J * w), -tau_max, tau_max)
        mujoco.mj_step(model, data); wmax = max(wmax, float(np.linalg.norm(A.omega_body(data))))
    q1 = A.attitude_dz(data); ach = float(np.linalg.norm(A.np_error_vector(q1, q0))); cmd = float(np.linalg.norm(tau_cmd))
    return q1, (ach / cmd if cmd > EXEC_CMD_FLOOR else None), wmax


def run_arm(m, Z0, actor, Vof, rho, dt, steps, Kp, Kd, tau_max, w0_radius, rng):
    d = mujoco.MjData(m); exits = conv = 0; rise = -np.inf; execs = []; finals = []; wmax = 0.0
    for z in Z0:
        mujoco.mj_resetData(m, d); A.set_attitude_dz(d, np.asarray(S.exp_quat(jp.asarray(z)), dtype=float))
        w0 = rng.normal(size=3); w0 *= w0_radius * rng.uniform() ** (1 / 3) / np.linalg.norm(w0)
        d.qvel[A.QVEL_ANG] = w0 if w0_radius > 0 else 0.0
        mujoco.mj_forward(m, d)
        Vp = Vof(z); exited = False
        for _ in range(steps):
            q1, tf, wm = step_outer_track(m, d, commanded_tau(actor, z), dt, Kp, Kd, tau_max)
            wmax = max(wmax, wm)
            if tf is not None and np.isfinite(tf): execs.append(tf)
            z = A.np_error_vector(q1, np.asarray(Q_GOAL, dtype=float)); Vn = Vof(z)
            exited |= Vn > rho; rise = max(rise, Vn - Vp); Vp = Vn
        exits += int(exited); finals.append(float(np.linalg.norm(z))); conv += int(np.linalg.norm(z) < 0.05)
    return dict(exits=exits, converged=conv, episodes=len(Z0), largest_V_increase=float(rise), omega_max=wmax,
                carried_out_median=float(np.median(execs)) if execs else None, carried_out_p5=float(np.percentile(execs, 5)) if execs else None,
                final_angle_median=float(np.median(finals)), final_angle_max=float(np.max(finals)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=900); ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--dt-outer", type=float, default=15.7)
    a = ap.parse_args(); t0 = time.time()
    m = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    J, tau_max = A.inertia_diag(m), IL.torque_limit(m); Kp, Kd, _, _ = IL.derive_gains(J, tau_max)
    actor = load_policy(a.seed, "ppo")
    SD._VJ, rho = SD.load_final()
    Z0 = SD.sample_in_Ds(None, None, rho, a.episodes, np.random.default_rng(1000 + a.seed))
    Vof = lambda z: float(np.asarray(SD.V_chart(None, None, jp.asarray(np.atleast_2d(z))), dtype=float).ravel()[0])
    res = {"nominal": run_arm(m, Z0, actor, Vof, rho, a.dt_outer, a.steps, Kp, Kd, tau_max, 0.0, np.random.default_rng(1))}
    wm = res["nominal"]["omega_max"]; print("nominal", json.dumps(res["nominal"]), flush=True)
    for name, frac, dt, ns in (("rate25", 0.25, a.dt_outer, a.steps), ("rate50", 0.5, a.dt_outer, a.steps), ("half", 0.0, a.dt_outer / 2, 2 * a.steps)):
        res[name] = dict(run_arm(m, Z0, actor, Vof, rho, dt, ns, Kp, Kd, tau_max, frac * wm, np.random.default_rng(2)),
                         initial_rate_radius=frac * wm, dt_outer=dt, steps=ns)
        print(name, json.dumps(res[name]), flush=True)
    out = dict(vehicle="CubeSat", seed=a.seed, rho=rho, episodes=a.episodes, steps=a.steps, dt_outer=a.dt_outer, Kp=Kp.tolist() if hasattr(Kp, "tolist") else Kp,
               arms=res, wall_s=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
