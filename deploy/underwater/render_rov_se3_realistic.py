"""Realistic underwater render of the BlueROV2 Heavy in MuJoCo under the PPO SE(3) policy (seed 0), as a GIF.

Usage: python underwater/render_rov_se3_realistic.py [--seed 0] [--out results/paper/rov_se3_realistic.gif]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hydro
import rov_se3_deploy_final as D
import jax.numpy as jp

import json as _json
_PARTS = _json.load(open(HERE / "assets/bluerov2_heavy/parts.json"))["parts"]
MESH_ASSETS = "".join(f'<mesh name="rov_{k}" file="{HERE}/assets/bluerov2_heavy/{p["file"]}" inertia="shell"/>'
                      f'<material name="rovmat_{k}" rgba="{p["rgba"][0]} {p["rgba"][1]} {p["rgba"][2]} 1" specular="0.35" shininess="0.6" reflectance="0.05"/>'
                      for k, p in enumerate(_PARTS)) + f'<mesh name="rov_hull" file="{HERE}/assets/bluerov2_heavy/hull.obj"/>' 
_EDGES = [((-0.235, -0.29, -0.118), (-0.235, -0.29, 0.142)), ((-0.235, -0.29, -0.118), (-0.235, 0.29, -0.118)), ((-0.235, -0.29, -0.118), (0.235, -0.29, -0.118)), ((-0.235, -0.29, 0.142), (-0.235, 0.29, 0.142)), ((-0.235, -0.29, 0.142), (0.235, -0.29, 0.142)), ((-0.235, 0.29, -0.118), (-0.235, 0.29, 0.142)), ((-0.235, 0.29, -0.118), (0.235, 0.29, -0.118)), ((-0.235, 0.29, 0.142), (0.235, 0.29, 0.142)), ((0.235, -0.29, -0.118), (0.235, -0.29, 0.142)), ((0.235, -0.29, -0.118), (0.235, 0.29, -0.118)), ((0.235, -0.29, 0.142), (0.235, 0.29, 0.142)), ((0.235, 0.29, -0.118), (0.235, 0.29, 0.142))]
def _ghost(mat):
    return "".join(f'<geom type="capsule" fromto="{p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]}" size="0.004" material="{mat}" '
                   f'contype="0" conaffinity="0" mass="0" group="2"/>' for p, q in _EDGES)
BODY_VIS = "".join(f'<geom type="mesh" mesh="rov_{k}" material="rovmat_{k}" contype="0" conaffinity="0" mass="0" group="1"/>'
                   for k in range(len(_PARTS))) + (
    '<light name="lamp_l" pos="0.21 0.13 0.0" dir="1 0 -0.35" cutoff="38" exponent="12" diffuse="1.0 0.95 0.85" '
    'specular="0.3 0.3 0.3" attenuation="1 0.25 0.08" castshadow="false"/>'
    '<light name="lamp_r" pos="0.21 -0.13 0.0" dir="1 0 -0.35" cutoff="38" exponent="12" diffuse="1.0 0.95 0.85" '
    'specular="0.3 0.3 0.3" attenuation="1 0.25 0.08" castshadow="false"/>')
SEA_Z, SEA_H = -2.45, 0.35
def _rocks_and_snow():
    from PIL import Image as _I
    hmap = np.asarray(_I.open(HERE / "assets/scene/seabed_h.png"), float) / 255.0
    r = np.random.default_rng(11); out = []
    for i in range(16):
        ang, rad = r.uniform(0, 2 * np.pi), r.uniform(1.0, 7.0); x, y = rad * np.cos(ang), rad * np.sin(ang)
        hz = hmap[int((1 - (y + 20) / 40) * (hmap.shape[0] - 1)), int((x + 20) / 40 * (hmap.shape[1] - 1))]
        sc = r.uniform(0.12, 0.45)
        out.append(f'<geom type="mesh" mesh="rock_{i % 4}" material="rock" pos="{x:.3f} {y:.3f} {SEA_Z + SEA_H * hz - 0.03 * sc:.3f}" '
                   f'euler="0 0 {r.uniform(0, 6.28):.3f}" mesh_scale_placeholder="{sc:.3f}" contype="0" conaffinity="0"/>')
    for i in range(0):
        x, y, z = r.uniform(-2.6, 2.6), r.uniform(-2.6, 2.6), r.uniform(-1.9, 1.6)
        out.append(f'<geom type="sphere" size="{r.uniform(0.005, 0.012):.4f}" pos="{x:.3f} {y:.3f} {z:.3f}" material="snow" '
                   f'contype="0" conaffinity="0" group="1"/>')
    return "".join(out)
WORLD_VIS = """
    <light name="sun" directional="true" pos="0 0 8" dir="0.15 0.25 -1" diffuse="0.52 0.68 0.72" specular="0.1 0.12 0.12" castshadow="true"/>
    <light name="scatter" directional="true" pos="0 0 -3" dir="-0.2 -0.1 1" diffuse="0.04 0.09 0.10" castshadow="false"/>
    <geom name="seabed" type="hfield" hfield="seabed" pos="0 0 {SEA_Z}" material="sand" contype="0" conaffinity="0"/>
    {ROCKS_SNOW}
    <body name="goal" pos="0 0 0">{GOAL_GHOST}
    </body>
    <body name="target" mocap="true" pos="0 0 0">{TARGET_GHOST}
    </body>
"""
ASSETS = """
  <asset>
    <texture name="water" type="skybox" builtin="gradient" rgb1="0.16 0.50 0.58" rgb2="0.0 0.04 0.07" width="512" height="512"/>
    <hfield name="seabed" file="{HERE}/assets/scene/seabed_h.png" size="20 20 {SEA_H} 0.1"/>
    <texture name="sandtex" type="2d" file="{HERE}/assets/scene/seabed_tex.png"/>
    <material name="sand" texture="sandtex" texrepeat="9 9" reflectance="0.0" specular="0.06" shininess="0.1"/>
    <material name="rock" rgba="0.30 0.29 0.26 1" specular="0.12" shininess="0.15"/>
    <material name="snow" rgba="0.80 0.90 0.88 1" emission="0.75" specular="0"/>
    <material name="wire_goal" rgba="0.25 1.0 0.5 1" emission="1" specular="0"/>
    <material name="wire_target" rgba="1.0 0.6 0.15 1" emission="1" specular="0"/>
    {ROCK_MESHES}
  </asset>
  <statistic extent="1" center="0 0 0"/>
  <visual><headlight ambient="0.14 0.21 0.23" diffuse="0.14 0.16 0.17" specular="0 0 0"/><quality shadowsize="4096"/>
          <map znear="0.01" zfar="60"/><global offwidth="960" offheight="720"/></visual>
"""


def build_model():
    xml = hydro.XML.read_text()
    xml = xml.replace('<geom name="hull"', BODY_VIS + '      <geom name="hull"', 1)
    xml = xml.replace('rgba="0.15 0.35 0.55 0.55"', 'rgba="0.15 0.35 0.55 0.0"', 1)
    xml = xml.replace('<light pos="0 0 2" dir="0 0 -1"/>', "", 1)
    rock_meshes = "".join(f'<mesh name="rock_{k}" file="{HERE}/assets/scene/rock_{k}.obj"/>' for k in range(4))
    assets = (ASSETS.strip().replace("{HERE}", str(HERE)).replace("{SEA_H}", str(SEA_H)).replace("{ROCK_MESHES}", rock_meshes)
              .replace("<asset>", "<asset>" + MESH_ASSETS, 1))
    xml = xml.replace("<worldbody>", assets + "\n  <worldbody>", 1)
    rs = _rocks_and_snow()
    import re as _re
    rs = _re.sub(r'mesh="rock_(\d)"([^>]*?)mesh_scale_placeholder="([0-9.]+)"', lambda mm: f'mesh="rock_{mm.group(1)}_{mm.group(3)}"{mm.group(2)}', rs)
    scaled = sorted(set(_re.findall(r'mesh="(rock_\d_[0-9.]+)"', rs)))
    xml = xml.replace("</asset>", "".join(f'<mesh name="{n}" file="{HERE}/assets/scene/rock_{n.split("_")[1]}.obj" scale="{n.split("_")[2]} {n.split("_")[2]} {n.split("_")[2]}"/>' for n in scaled) + "</asset>", 1)
    world = (WORLD_VIS.replace("{SEA_Z}", str(SEA_Z)).replace("{ROCKS_SNOW}", rs)
             .replace("{GOAL_GHOST}", _ghost("wire_goal")).replace("{TARGET_GHOST}", _ghost("wire_target")))
    xml = xml.replace("</worldbody>", world + "  </worldbody>", 1)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.body("rov").id == 1
    m.vis.map.fogstart, m.vis.map.fogend = 0.8, 8.5
    m.vis.rgba.fog[:] = [0.03, 0.20, 0.25, 1.0]; m.vis.rgba.haze[:] = [0.05, 0.27, 0.32, 1.0]
    return m, mujoco.MjData(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dt-outer", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=3.0, help="simulated seconds per second of GIF")
    ap.add_argument("--out", default=str(HERE.parent / "results/paper/rov_se3_realistic.gif"))
    a = ap.parse_args()

    G, amax, cfg, V, tau_of, rec = D.setup(a.seed); rho = rec["rho"]
    rng = np.random.default_rng(3)
    Z = D.sample_starts(V, rho, cfg.radius, 4000, rng)
    rot, tra = np.linalg.norm(Z[:, :3], axis=1), np.linalg.norm(Z[:, 3:], axis=1)
    z0 = Z[np.argmin((rot - 2.0) ** 2 + (tra - 1.3) ** 2)]
    m, d = build_model()
    inner = hydro.PoseInnerLoop(m, kp_pos=400.0, kd_pos=160.0)
    nsub = int(round(a.dt_outer / m.opt.timestep))
    every = max(1, int(round(a.speed / a.fps / m.opt.timestep)))
    tid = m.body("target").mocapid[0]

    ren = mujoco.Renderer(m, 540, 720)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1; ren.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.1; cam.azimuth = 140; cam.elevation = -12
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 22)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()

    X = np.asarray(G.Exp(jp.asarray(z0[None]))); mujoco.mj_resetData(m, d); D.set_state(d, X); mujoco.mj_forward(m, d)
    cam.lookat[:] = 0.75 * d.qpos[:3]
    frames = []

    def snap(k, cost, v):
        cam.lookat[:] = 0.85 * cam.lookat + 0.15 * (0.75 * d.qpos[:3])
        ren.update_scene(d, camera=cam); img = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, 720, 72], fill=(0, 10, 20, 150)); dr.rectangle([0, 500, 720, 540], fill=(0, 10, 20, 150))
        dr.text((18, 14), "BlueROV2 in MuJoCo, PPO policy on SE(3)", font=font, fill=(235, 240, 245))
        dr.text((18, 44), f"step {k} of {a.steps}   t = {d.time:5.1f} s   pose error {cost:.3f}   V/ρ = {v:.3f}", font=small, fill=(220, 228, 235))
        dr.text((18, 509), "green: goal pose      orange: target pose for the current step", font=small, fill=(200, 210, 220))
        frames.append(img)

    cost0 = D.chart_norm(G, X); snap(0, cost0, float(np.asarray(V(G.Log(jp.asarray(X))))[0]) / rho)
    for k in range(a.steps):
        tau = np.asarray(tau_of(jp.asarray(X)))
        Xd = np.asarray(G.compose(jp.asarray(X), G.Exp(jp.asarray(tau))))
        R_des, p_des = D.pose(Xd)
        d.mocap_pos[tid] = p_des
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(R_des).as_quat(); d.mocap_quat[tid] = [q[3], q[0], q[1], q[2]]
        cost = D.chart_norm(G, X); v = float(np.asarray(V(G.Log(jp.asarray(X))))[0]) / rho
        for i in range(nsub):
            d.xfrc_applied[1, :] = 0
            hydro.apply_hydro(m, d)
            d.ctrl[:] = inner(d, R_des, p_des)
            hydro.apply_added_mass(m, d)
            mujoco.mj_step(m, d)
            if i % every == 0: snap(k, cost, v)
        X = D.X_of(d)
    for _ in range(a.fps): snap(a.steps, D.chart_norm(G, X), float(np.asarray(V(G.Log(jp.asarray(X))))[0]) / rho)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, start |omega| {np.linalg.norm(z0[:3]):.2f} rad, |v| {np.linalg.norm(z0[3:]):.2f} m, "
          f"final ||Log E|| {D.chart_norm(G, X):.2e}")


if __name__ == "__main__":
    main()
