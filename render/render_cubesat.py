"""CubeSat attitude manoeuvre under the PPO policy on SO(3), rendered for the project page."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

DZ = Path(os.environ.get("DZ_SO3", Path(__file__).resolve().parents[1] / "deploy"))
sys.path[:0] = [str(DZ), str(DZ / "spacecraft")]

import jax.numpy as jp
import mujoco
from PIL import Image

import adapter as A
import inner_loop as IL
import render_cubesat_gif as RC
import sc_deploy_study_final as SD
from dz_so3 import so3 as S
from sc_nu_measure_l4dc import Q_GOAL, commanded_tau, load_policy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post

W, H = 1280, 720


def model():
    xml = RC.build_xml().replace('offwidth="960" offheight="720"', f'offwidth="{W}" offheight="{H}"')
    atm = ('<geom name="atmosphere" type="sphere" size="61.2" pos="-25 75 -75" rgba="0.35 0.62 1.0 0.16" '
           'contype="0" conaffinity="0" group="1"/>')
    xml = xml.replace('<body name="goal"', atm + '\n    <body name="goal"', 1)
    return mujoco.MjModel.from_xml_string(xml)


def start(Vj, rho, rng):
    z = rng.normal(size=(20000, 3))
    z *= (rng.uniform(1.9, 2.9, (len(z), 1)) / np.linalg.norm(z, axis=1, keepdims=True))
    v = np.asarray(Vj(jp.asarray(z))).ravel()
    ok = np.where(v <= 0.9 * rho)[0]
    return z[ok[np.argmax(np.linalg.norm(z[ok], axis=1))]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dt-outer", type=float, default=15.7)
    ap.add_argument("--frame-dt", type=float, default=0.75)
    ap.add_argument("--out", default="docs/assets/media/cubesat")
    a = ap.parse_args()

    Vj, rho = SD.load_final()
    z = start(Vj, rho, np.random.default_rng(7))
    m = model()
    d = mujoco.MjData(m)
    J, tau_max = A.inertia_diag(m), IL.torque_limit(m)
    Kp, Kd, _, _ = IL.derive_gains(J, tau_max)
    actor = load_policy(a.seed)
    mujoco.mj_resetData(m, d)
    A.set_attitude_dz(d, np.asarray(S.exp_quat(jp.asarray(z)), float))
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0
    opt = mujoco.MjvOption()
    grade = post.Grade(W, H, vignette=0.6, bloom=0.45, thresh=190, tint=(1.0, 0.98, 1.04))
    every = max(1, int(round(a.frame_dt / m.opt.timestep)))
    n_in = int(round(a.dt_outer / m.opt.timestep))
    total = a.steps * n_in // every
    hero, hud, k = [], [], 0

    def snap(step):
        s = len(hero) / max(1, total)
        cam.distance = 0.78 - 0.16 * s
        cam.azimuth = 128 + 46 * s
        cam.elevation = -8 - 10 * s
        v = float(np.asarray(Vj(jp.asarray(A.np_error_vector(A.attitude_dz(d), np.asarray(Q_GOAL, float))[None]))).ravel()[0])
        err = np.degrees(np.linalg.norm(A.np_error_vector(A.attitude_dz(d), np.asarray(Q_GOAL, float))))
        opt.geomgroup[2] = 0
        r.update_scene(d, camera=cam, scene_option=opt)
        hero.append(grade(r.render()))
        opt.geomgroup[2] = 1
        r.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(grade(r.render()))
        post.hud(img, "3U CubeSat  ·  attitude on SO(3)",
                 [("policy step", f"{step} / {a.steps}"), ("attitude error", f"{err:6.1f}°"), ("V / ρ", f"{v / rho:6.3f}")], v / rho)
        hud.append(np.asarray(img))

    snap(0)
    for step in range(1, a.steps + 1):
        tau_cmd = commanded_tau(actor, z)
        q0 = A.attitude_dz(d)
        q_des = np.asarray(S.quat_mul(jp.asarray(q0), jp.asarray(np.asarray(S.exp_quat(jp.asarray(tau_cmd))))), float)
        for _ in range(n_in):
            q = A.attitude_dz(d)
            e_R = A.np_error_vector(q, q_des)
            w = A.omega_body(d)
            d.ctrl[:] = np.clip(-Kp * e_R - Kd * w + np.cross(w, J * w), -tau_max, tau_max)
            mujoco.mj_step(m, d)
            k += 1
            if k % every == 0:
                snap(step)
        z = A.np_error_vector(A.attitude_dz(d), np.asarray(Q_GOAL, float))
    post.write_video(post.fade(hero), a.out + "_hero")
    post.write_video(post.fade(hud), a.out + "_hud", gif=True)
    print(f"{len(hero)} frames, start |z| {np.linalg.norm(start(Vj, rho, np.random.default_rng(7))):.2f} rad, "
          f"final error {np.degrees(np.linalg.norm(z)):.3f} deg")


if __name__ == "__main__":
    main()
