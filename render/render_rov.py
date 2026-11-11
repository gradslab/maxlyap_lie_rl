"""BlueROV2 pose manoeuvre under the PPO policy on SE(3), rendered for the project page."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

DZ = Path(os.environ.get("DZ_SO3", Path(__file__).resolve().parents[1] / "deploy"))
sys.path[:0] = [str(DZ), str(DZ / "underwater")]

import jax.numpy as jp
import mujoco
from PIL import Image
from scipy.spatial.transform import Rotation

import hydro
import render_rov_se3_realistic as RR
import rov_se3_deploy_final as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post

W, H = 1280, 720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dt-outer", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=2.4)
    ap.add_argument("--out", default="docs/assets/media/rov")
    a = ap.parse_args()

    G, amax, cfg, V, tau_of, rec = D.setup(a.seed)
    rho = rec["rho"]
    Z = D.sample_starts(V, rho, cfg.radius, 4000, np.random.default_rng(3))
    rot, tra = np.linalg.norm(Z[:, :3], axis=1), np.linalg.norm(Z[:, 3:], axis=1)
    z0 = Z[np.argmin((rot - 2.0) ** 2 + (tra - 1.3) ** 2)]

    RR.ASSETS = RR.ASSETS.replace('offwidth="960" offheight="720"', f'offwidth="{W}" offheight="{H}"')
    m, d = RR.build_model()
    inner = hydro.PoseInnerLoop(m, kp_pos=400.0, kd_pos=160.0)
    nsub = int(round(a.dt_outer / m.opt.timestep))
    every = max(1, int(round(a.speed / a.fps / m.opt.timestep)))
    tid = m.body("target").mocapid[0]
    ren = mujoco.Renderer(m, H, W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    opt = mujoco.MjvOption()
    grade = post.Grade(W, H, vignette=0.7, bloom=0.3, thresh=170, tint=(0.92, 1.02, 1.04), lift=(0, 6, 9))
    snow = post.Snow(W, H, seed=4)
    total = a.steps * nsub // every

    X = np.asarray(G.Exp(jp.asarray(z0[None])))
    mujoco.mj_resetData(m, d)
    D.set_state(d, X)
    mujoco.mj_forward(m, d)
    cam.lookat[:] = 0.75 * d.qpos[:3]
    hero, hud = [], []

    def snap(k, cost, v):
        s = len(hero) / max(1, total)
        cam.lookat[:] = 0.88 * cam.lookat + 0.12 * (0.7 * d.qpos[:3])
        cam.distance = 2.3 - 0.5 * s
        cam.azimuth = 128 + 55 * s
        cam.elevation = -9 - 8 * s
        t = len(hero) / a.fps
        rays = post.god_rays(W, H, t)
        out = []
        for g in (0, 1):
            opt.geomgroup[2] = g
            ren.update_scene(d, camera=cam, scene_option=opt)
            f = grade(np.clip(ren.render().astype(np.float32) + rays, 0, 255).astype(np.uint8))
            out.append(f)
        hero.append(np.asarray(snow(Image.fromarray(out[0]), t)))
        img = Image.fromarray(out[1])
        post.hud(img, "BlueROV2  ·  pose on SE(3)",
                 [("policy step", f"{k} / {a.steps}"), ("pose error", f"{cost:7.3f}"), ("V / ρ", f"{v:7.3f}")], v)
        hud.append(np.asarray(img))

    Vr = lambda X: float(np.asarray(V(G.Log(jp.asarray(X))))[0]) / rho
    snap(0, D.chart_norm(G, X), Vr(X))
    for k in range(a.steps):
        tau = np.asarray(tau_of(jp.asarray(X)))
        Xd = np.asarray(G.compose(jp.asarray(X), G.Exp(jp.asarray(tau))))
        R_des, p_des = D.pose(Xd)
        d.mocap_pos[tid] = p_des
        q = Rotation.from_matrix(R_des).as_quat()
        d.mocap_quat[tid] = [q[3], q[0], q[1], q[2]]
        cost, v = D.chart_norm(G, X), Vr(X)
        for i in range(nsub):
            d.xfrc_applied[1, :] = 0
            hydro.apply_hydro(m, d)
            d.ctrl[:] = inner(d, R_des, p_des)
            hydro.apply_added_mass(m, d)
            mujoco.mj_step(m, d)
            if i % every == 0:
                snap(k + 1, cost, v)
        X = D.X_of(d)
    post.write_video(post.fade(hero), a.out + "_hero")
    post.write_video(post.fade(hud), a.out + "_hud", gif=True)
    print(f"{len(hero)} frames, start rotation {np.linalg.norm(z0[:3]):.2f} rad, translation {np.linalg.norm(z0[3:]):.2f} m, "
          f"final error {D.chart_norm(G, X):.2e}")


if __name__ == "__main__":
    main()
