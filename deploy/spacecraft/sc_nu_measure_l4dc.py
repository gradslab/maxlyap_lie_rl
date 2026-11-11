"""nu measurement: how far the RIGID BODY departs from the CERTIFIED KINEMATIC MAP in one outer step.

Usage: python spacecraft/sc_nu_measure.py <seed> [--tag ppo] [--dt-outer 4.0] [--n 4000]
Writes spacecraft/results/sc_nu_seed<seed>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import mujoco
import jax.numpy as jp
from dz_so3 import so3 as S, env as E
from dz_so3.nets import mlp_apply
from dz_so3.ppo import AgentParams
from dz_so3.certificate import CertConfig, CertParams, V_chart
import adapter as A
import inner_loop as IL

from gamma_const import GAMMA_CERTIFIED as GAMMA
STEP_LEN = 0.1
Q_GOAL = jp.asarray([0.0, 0.0, 0.0, 1.0])


def load_policy(seed, tag="ppo"):
    """L4DC variant: frozen actor with the equilibrium offset applied AFTER the output tanh,"""
    with open(ROOT / f"results/ppo_step0.1_error_matrix/agent_seed{seed}.pkl", "rb") as f:
        agent = AgentParams(*pickle.load(f))
    actor = [(jp.asarray(w), jp.asarray(b)) for w, b in agent.actor]
    obs_g = E.OBS_FNS["error_matrix"](Q_GOAL[None], Q_GOAL[None])
    bias = jp.tanh(mlp_apply(actor, obs_g, jp.tanh)[0])
    return (actor, bias)


def load_cert(seed, tag):
    with open(ROOT / f"results/e1_seed{seed}_step0.1_{tag}/cert_params.pkl", "rb") as f:
        saved = pickle.load(f)
    p = CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in saved[0]],
                   alpha_raw=jp.asarray(saved[1]), eps_raw=jp.asarray(saved[2]))
    return p, CertConfig()


def commanded_tau(actor, z):
    """Policy action at z with the post-tanh offset, clamped as in lie_rl.env.clip_tau."""
    actor, bias = actor
    q = S.exp_quat(jp.asarray(z))
    obs = E.obs_error_matrix(q[None], Q_GOAL[None])
    a = jp.tanh(mlp_apply(actor, obs, jp.tanh)[0]) - bias
    return np.array(E._clamped_tau(a, STEP_LEN), dtype=float, copy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int); ap.add_argument("--tag", default="ppo")
    ap.add_argument("--dt-outer", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()

    gate = HERE / "results/sc_validate.json"
    if not gate.exists() or not json.load(open(gate)).get("ALL_PASSED"):
        raise SystemExit("REFUSING TO RUN: spacecraft/sc_validate.py has not passed. An unvalidated "
                         "adapter yields a confident, plausible, wrong nu.")

    m = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    J, tau_max = A.inertia_diag(m), IL.torque_limit(m)
    Kp, Kd, wn, t_slew = IL.derive_gains(J, tau_max)
    if a.dt_outer <= t_slew.max():
        raise SystemExit(f"dt_outer={a.dt_outer} <= max slew time {t_slew.max():.3f}s: the commanded "
                         f"increment is not physically executable; nu would measure that, not dynamics.")

    actor = load_policy(a.seed, a.tag)
    cp, cfg = load_cert(a.seed, a.tag)
    rho = json.load(open(ROOT / f"crown_ext/results/crown_sound_rho_seed{a.seed}.json"))["rho_sound"]

    rng = np.random.default_rng(a.seed)
    z = rng.normal(size=(a.n, 3))
    z = z / np.linalg.norm(z, axis=1, keepdims=True) * (rng.uniform(0, 1, (a.n, 1)) ** (1 / 3)) * GAMMA
    V = np.asarray(V_chart(cp, cfg, jp.asarray(z)), dtype=float).ravel()
    in_Ds = V <= rho

    rows = []
    d = mujoco.MjData(m)
    for i in range(a.n):
        zi = z[i]
        tau = commanded_tau(actor, zi)
        qz = np.asarray(S.exp_quat(jp.asarray(zi)), dtype=float)
        z_model = np.asarray(S.log_quat(S.quat_mul(jp.asarray(qz),
                                                   jp.asarray(np.asarray(S.exp_quat(jp.asarray(tau)))))),
                             dtype=float)
        mujoco.mj_resetData(m, d)
        A.set_attitude_dz(d, qz)
        d.qvel[A.QVEL_ANG] = 0.0
        mujoco.mj_forward(m, d)
        out = IL.step_outer(m, d, tau, a.dt_outer, Kp, Kd, tau_max)
        z_plant = A.error_vector(out["q1"], np.asarray(Q_GOAL, dtype=float))
        nu = float(np.linalg.norm(A.error_vector(
            np.asarray(S.exp_quat(jp.asarray(z_plant)), dtype=float),
            np.asarray(S.exp_quat(jp.asarray(z_model)), dtype=float))))
        rows.append(dict(theta=float(np.linalg.norm(zi)), V=float(V[i]), in_Ds=bool(in_Ds[i]),
                         cmd=out["commanded_norm"], achieved=out["achieved_norm"],
                         track=out["tracking_fraction"], resid_rate=out["residual_rate_norm"],
                         nu=nu))

    def stats(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return dict(max=float(v.max()), p99=float(np.percentile(v, 99)),
                    median=float(np.median(v)), min=float(v.min())) if v.size else None

    sel = [r for r in rows if r["in_Ds"]]
    out = dict(
        branch="SC-1 isolated; EMPIRICAL out-of-model study (G3), NOT a certification claim",
        seed=a.seed, tag=a.tag, dt_outer=a.dt_outer, alpha_max=IL.ALPHA_MAX,
        inertia=J.tolist(), tau_max=tau_max, Kp=Kp.tolist(), Kd=Kd.tolist(),
        t_slew_per_axis=t_slew.tolist(), rho_sound=rho,
        n_sampled=a.n, n_in_Ds=len(sel),
        initial_rate="omega = 0 at every grid point (the kinematic model carries no rate; this is the "
                     "BEST CASE for the model)",
        nu_all=stats([r["nu"] for r in rows]),
        nu_in_Ds=stats([r["nu"] for r in sel]),
        tracking_fraction_in_Ds=stats([r["track"] for r in sel]),
        residual_rate_in_Ds=stats([r["resid_rate"] for r in sel]),
        parameter_provenance="3U CubeSat per the CubeSat Design Specification (10x10x34 cm, 4.0 kg); inertia is a STATED uniform-rectangular-prism assumption (derives 0.0419/0.0067, simulated 0.049/0.008, and the inertia stress arm spans 0.35x-2.8x so it contains the derivation at 0.85x); reaction-wheel torque 1 mN.m CONFIRMED 2026-08-24 against RL-RW-0.01-Data-Sheet.pdf in the repo ('TORQUE +/-1 mNm at 0.01 Nms'). The earlier PLACEHOLDER blocker is discharged: 0.01 Nms is the wheel's stored MOMENTUM, not its torque.")
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / f"results/sc_nu_seed{a.seed}.json", "w"), indent=2)
    print(f"  seed {a.seed}  dt_outer={a.dt_outer}s  n={a.n} ({len(sel)} in D_s)")
    for k in ("nu_all", "nu_in_Ds", "tracking_fraction_in_Ds", "residual_rate_in_Ds"):
        print(f"    {k:26s} {out[k]}")


if __name__ == "__main__":
    main()
