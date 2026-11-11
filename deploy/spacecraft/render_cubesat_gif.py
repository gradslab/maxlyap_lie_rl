"""Render the 3U CubeSat in MuJoCo under the PPO SO(3) policy and its reaction-wheel inner loop, as a GIF.

Usage: python spacecraft/render_cubesat_gif.py [--z0 2.0 -1.2 0.8] [--steps 22] [--out results/paper/cubesat_mirata.gif]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import jax.numpy as jp
from dz_so3 import so3 as S
import adapter as A
import inner_loop as IL
from sc_nu_measure_l4dc import load_policy, commanded_tau, Q_GOAL

AS = HERE / "assets"; MI = AS / "mirata"
PARTS = json.load(open(MI / "parts.json"))["parts"]
LOOK = {"solar": 'specular="0.65" shininess="0.92" reflectance="0.18"',
        "gold": 'specular="0.85" shininess="0.75" reflectance="0.30"',
        "metal": 'specular="0.70" shininess="0.70" reflectance="0.15"',
        "copper": 'specular="0.70" shininess="0.65" reflectance="0.20"',
        "dark": 'specular="0.20" shininess="0.30" reflectance="0.02"'}


def build_xml():
    xml = (HERE / "model.xml").read_text()
    assets = "<asset>\n" + "".join(
        f'    <mesh name="sat_{k}" file="{MI / p["file"]}"/>\n'
        f'    <material name="satmat_{k}" rgba="{p["rgba"][0]} {p["rgba"][1]} {p["rgba"][2]} 1" {LOOK[p["kind"]]}/>\n'
        for k, p in enumerate(PARTS)) + f"""    <texture name="sky" type="skybox" file="{AS / 'stars.png'}" gridsize="3 4" gridlayout=".U..LFRB.D.."/>
    <texture name="earthtex" type="2d" file="{AS / 'earth.png'}"/>
    <material name="earth" texture="earthtex" specular="0.15" shininess="0.2" emission="0.05"/>
    <mesh name="earth" file="{AS / 'earth_sphere.obj'}" scale="60 60 60"/>
  </asset>
  <statistic extent="1" center="0 0 0"/>   <!-- znear/zfar are fractions of the extent: without this the large Earth sets it -->
  <visual><headlight ambient="0.11 0.11 0.13" diffuse="0.05 0.05 0.05" specular="0 0 0"/><quality shadowsize="4096"/>
          <map znear="0.005" zfar="400" shadowclip="1.5"/><global offwidth="960" offheight="720"/></visual>
"""
    vis = "".join(f'      <geom type="mesh" mesh="sat_{k}" material="satmat_{k}" contype="0" conaffinity="0" group="1"/>\n'
                  for k in range(len(PARTS)))
    xml = xml.replace('<geom name="bus" type="box" size="0.05 0.05 0.17" rgba="0.6 0.6 0.65 1"',
                      vis + '      <geom name="bus" type="box" size="0.05 0.05 0.17" rgba="0.6 0.6 0.65 0"', 1)
    world = """
    <light name="sun" directional="true" pos="0 0 5" dir="-0.35 0.85 -0.4" diffuse="1.0 0.97 0.92" specular="0.6 0.6 0.6" castshadow="true"/>
    <light name="earthshine" directional="true" pos="0 0 -5" dir="0.25 -0.7 0.67" diffuse="0.16 0.2 0.28" specular="0.05 0.05 0.08" castshadow="false"/>
    <geom name="earth" type="mesh" mesh="earth" material="earth" pos="-25 75 -75" euler="0.4 0.3 2.2" contype="0" conaffinity="0"/>
    <body name="goal" pos="0 0 0"><geom type="capsule" fromto="-0.056 -0.056 -0.176 -0.056 -0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 -0.056 -0.176 -0.056 0.056 -0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 -0.056 -0.176 0.056 -0.056 -0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 -0.056 0.176 -0.056 0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 -0.056 0.176 0.056 -0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 0.056 -0.176 -0.056 0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 0.056 -0.176 0.056 0.056 -0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="-0.056 0.056 0.176 0.056 0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="0.056 -0.056 -0.176 0.056 -0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="0.056 -0.056 -0.176 0.056 0.056 -0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="0.056 -0.056 0.176 0.056 0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/><geom type="capsule" fromto="0.056 0.056 -0.176 0.056 0.056 0.176" size="0.0012" rgba="0.3 1.0 0.55 1" contype="0" conaffinity="0" group="2"/></body>
    <camera name="view" pos="0.62 -0.78 0.30" xyaxes="0.78 0.62 0 -0.17 0.21 0.96" fovy="38"/>
"""
    xml = xml.replace("<worldbody>", assets + "\n  <worldbody>", 1)
    xml = xml.replace("</worldbody>", world + "  </worldbody>", 1)
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--z0", type=float, nargs=3, default=[2.0, -1.2, 0.8])
    ap.add_argument("--steps", type=int, default=22)
    ap.add_argument("--dt-outer", type=float, default=15.7)
    ap.add_argument("--frame-dt", type=float, default=1.2)
    ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", default=str(ROOT / "results/paper/cubesat_mirata.gif"))
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_string(build_xml()); d = mujoco.MjData(m)
    ref = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    b = m.body("sc").id; br = ref.body("sc").id
    assert np.allclose(m.body_inertia[b], ref.body_inertia[br]) and np.isclose(m.body_mass[b], ref.body_mass[br]), "visual geoms changed the dynamics"
    J, tau_max = A.inertia_diag(m), IL.torque_limit(m)
    assert m.body("sc").id == 1 and np.allclose(J, A.inertia_diag(ref)) and tau_max == IL.torque_limit(ref), "helpers read the wrong body"
    Kp, Kd, _, t_slew = IL.derive_gains(J, tau_max)
    actor = load_policy(a.seed)
    z = np.asarray(a.z0, float)
    mujoco.mj_resetData(m, d); A.set_attitude_dz(d, np.asarray(S.exp_quat(jp.asarray(z)), float)); d.qvel[A.QVEL_ANG] = 0.0
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, a.height, a.width); cam = m.camera("view").id
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    frames, t, every = [], 0.0, max(1, int(round(a.frame_dt / m.opt.timestep)))
    n = int(round(a.dt_outer / m.opt.timestep)); k_total = 0

    def snap(step):
        r.update_scene(d, camera=cam); img = Image.fromarray(r.render())
        err = np.degrees(np.linalg.norm(A.np_error_vector(A.attitude_dz(d), np.asarray(Q_GOAL, float))))
        dr = ImageDraw.Draw(img)
        dr.text((16, 12), "3U CubeSat in MuJoCo, PPO policy on SO(3)", fill=(235, 235, 235), font=font)
        dr.text((16, 38), f"step {step} of {a.steps}   t = {t:5.1f} s   attitude error {err:.1f}°", fill=(235, 235, 235), font=font)
        dr.text((16, a.height - 32), "green: goal attitude", fill=(120, 230, 150), font=font)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=200))

    snap(0)
    for step in range(1, a.steps + 1):
        tau_cmd = commanded_tau(actor, z)
        q0 = A.attitude_dz(d)
        q_des = np.asarray(S.quat_mul(jp.asarray(q0), jp.asarray(np.asarray(S.exp_quat(jp.asarray(tau_cmd))))), float)
        for _ in range(n):
            q = A.attitude_dz(d); e_R = A.np_error_vector(q, q_des); w = A.omega_body(d)
            d.ctrl[:] = np.clip(-Kp * e_R - Kd * w + np.cross(w, J * w), -tau_max, tau_max)
            mujoco.mj_step(m, d); t += m.opt.timestep; k_total += 1
            if k_total % every == 0: snap(step)
        z = A.np_error_vector(A.attitude_dz(d), np.asarray(Q_GOAL, float))
    for _ in range(25): frames.append(frames[-1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
    print(f"wrote {a.out}: {len(frames)} frames, final attitude error {np.degrees(np.linalg.norm(z)):.3f} deg, t_slew {t_slew.max():.2f} s")


if __name__ == "__main__":
    main()
