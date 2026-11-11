"""Deployment study: roll the certified policy on the MuJoCo rigid body, with a positive control.

Usage: python spacecraft/sc_deploy_study.py <seed> [--tag ppo] [--episodes 300] [--steps 60]
Writes spacecraft/results/sc_deploy_seed<seed>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import mujoco
import jax.numpy as jp
from dz_so3 import so3 as S
import pickle
FINAL = ROOT.parent / "lie_rl" / "results" / "final"
FINAL_ID = "SO3__ppo_SO3_ls-1.5__s0__base"
_VJ = None


def load_final():
    """the final anchored candidate (lie_rl) on the rotation-vector chart, and its level"""
    sys.path.insert(0, str(ROOT.parent / "lie_rl"))
    import jax
    from lie_rl import certificate as LC, anchor as AN
    from lie_rl.groups import GROUPS
    cert = pickle.load(open(FINAL / "cert" / f"{FINAL_ID}.pkl", "rb")); assert AN.enable_from_cert(cert)
    cfg = LC.CertConfig(**cert["cfg"])
    p = LC.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                      alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
    ell = LC.make_ell(GROUPS["SO3"])
    rho = json.load(open(FINAL / "eval" / f"{FINAL_ID}.json"))["fns"]["V"]["level"]
    return jax.jit(lambda z: LC.V_chart(p, cfg, ell, z)), rho


def V_chart(cp, cfg, z):
    return _VJ(jp.atleast_2d(jp.asarray(z)))
import adapter as A
import inner_loop as IL
from sc_nu_measure_l4dc import load_policy, load_cert, commanded_tau, GAMMA, Q_GOAL


EXEC_CMD_FLOOR = 0.01 * (3.141592653589793 * 0.1)


def sample_in_Ds(cp, cfg, rho, n, rng, band=None):
    """Rejection-sample initial conditions inside D_rho."""
    out = []
    while len(out) < n:
        z = rng.normal(size=(4 * n, 3))
        z = z / np.linalg.norm(z, axis=1, keepdims=True) * (rng.uniform(0, 1, (4 * n, 1)) ** (1 / 3)) * GAMMA
        V = np.asarray(V_chart(cp, cfg, jp.asarray(z)), dtype=float).ravel()
        keep = (V <= rho) if band is None else ((V >= band[0] * rho) & (V <= band[1] * rho))
        out.extend(list(z[keep]))
    return np.array(out[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int); ap.add_argument("--tag", default="ppo")
    ap.add_argument("--trace", default=None,
                    help="also write per-step theta/V traces to this .npz "
                         "(figure data only; the JSON summary is byte-identical without it)")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dt-outer", type=float, default=4.0)
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    metavar=("LO","HI"), help="restrict V/rho to [LO,HI] (boundary stress)")
    a = ap.parse_args()

    gate = HERE / "results/sc_validate.json"
    if not gate.exists() or not json.load(open(gate)).get("ALL_PASSED"):
        raise SystemExit("REFUSING TO RUN: sc_validate.py has not passed.")

    m = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    J, tau_max = A.inertia_diag(m), IL.torque_limit(m)
    Kp, Kd, _, t_slew = IL.derive_gains(J, tau_max)
    actor = load_policy(a.seed, a.tag)
    global _VJ
    _VJ, rho = load_final(); cp = cfg = None
    r_origin = json.load(open(ROOT / f"results/origin_closure_seed{a.seed}_error_matrix.json"))["r_origin"]

    rng = np.random.default_rng(1000 + a.seed)
    Z0 = sample_in_Ds(cp, cfg, rho, a.episodes, rng, band=a.band)

    def Vof(z):
        return float(np.asarray(V_chart(cp, cfg, jp.asarray(np.atleast_2d(z))), dtype=float).ravel()[0])

    c_exit = c_rise = c_core = 0
    c_worst_rise = 0.0
    TR = {k: np.full((a.episodes, a.steps + 1), np.nan) for k in ("c_th", "c_V", "p_th", "p_V")}
    for i in range(a.episodes):
        z = Z0[i].copy(); Vp = Vof(z); exited = False
        TR["c_th"][i, 0] = np.linalg.norm(z); TR["c_V"][i, 0] = Vp
        for k in range(a.steps):
            tau = commanded_tau(actor, z)
            q = np.asarray(S.exp_quat(jp.asarray(z)), dtype=float)
            z = A.np_log_quat(A.np_quat_mul(q, A.np_exp_quat(tau)))
            Vn = Vof(z)
            if Vn > rho: exited = True
            c_worst_rise = max(c_worst_rise, Vn - Vp)
            if Vn - Vp > 0: c_rise += 1
            TR["c_th"][i, k + 1] = np.linalg.norm(z); TR["c_V"][i, k + 1] = Vn
            Vp = Vn
        c_exit += int(exited)
        c_core += int(np.linalg.norm(z) <= r_origin)
    control_ok = (c_exit == 0)

    d = mujoco.MjData(m)
    p_exit = p_core = 0
    p_worst_rise = 0.0; finals = []; rises = 0; total = 0
    execs = []
    for i in range(a.episodes):
        z = Z0[i].copy()
        mujoco.mj_resetData(m, d)
        A.set_attitude_dz(d, np.asarray(S.exp_quat(jp.asarray(z)), dtype=float))
        d.qvel[A.QVEL_ANG] = 0.0
        mujoco.mj_forward(m, d)
        Vp = Vof(z); exited = False
        TR["p_th"][i, 0] = np.linalg.norm(z); TR["p_V"][i, 0] = Vp
        for k in range(a.steps):
            tau = commanded_tau(actor, z)
            out = IL.step_outer(m, d, tau, a.dt_outer, Kp, Kd, tau_max)
            if float(np.linalg.norm(tau)) > EXEC_CMD_FLOOR:
                _tf = out.get("tracking_fraction")
                if _tf is not None and np.isfinite(_tf):
                    execs.append(float(_tf))
            z = A.np_error_vector(out["q1"], np.asarray(Q_GOAL, dtype=float))
            Vn = Vof(z); total += 1
            if Vn > rho: exited = True
            if Vn - Vp > 0: rises += 1
            p_worst_rise = max(p_worst_rise, Vn - Vp)
            TR["p_th"][i, k + 1] = np.linalg.norm(z); TR["p_V"][i, k + 1] = Vn
            Vp = Vn
        p_exit += int(exited)
        p_core += int(np.linalg.norm(z) <= r_origin)
        finals.append(float(np.linalg.norm(z)))

    finals = np.array(finals)
    res = dict(
        branch="SC-1 isolated; EMPIRICAL out-of-model study (G3), NOT a certification claim",
        seed=a.seed, tag=a.tag, episodes=a.episodes, steps=a.steps, dt_outer=a.dt_outer,
        band=a.band, band_note=("uniform in D_rho" if a.band is None else
                                f"BOUNDARY SHELL V/rho in {a.band} -- where invariance is stressed"),
        rho_sound=rho, r_origin=r_origin,
        inertia=list(A.inertia_diag(m)), tau_max=IL.torque_limit(m),
        theorem_scope="The certificate proves invariance of D_rho for the KINEMATIC map only. "
                      "nu/nu* = 1.3e7 here, so the theorem makes NO prediction about this plant; "
                      "exits are not certificate failures and are not scored as such.",
        control_kinematic=dict(exits=c_exit, exit_fraction=c_exit / a.episodes,
                               reached_origin_core=c_core, worst_V_rise=c_worst_rise,
                               PASSED=bool(control_ok),
                               note="MUST be zero exits; this validates the harness"),
        plant=dict(executed_fraction=dict(median=float(np.median(execs)),
                                          min=float(np.min(execs)),
                                          max=float(np.max(execs)),
                                          n=int(len(execs)))),
        plant_mujoco=dict(exits=p_exit, exit_fraction=p_exit / a.episodes,
                          reached_origin_core=p_core,
                          core_fraction=p_core / a.episodes,
                          steps_with_V_rise=rises, steps_total=total,
                          V_rise_fraction=rises / max(total, 1),
                          worst_V_rise=p_worst_rise,
                          final_theta=dict(max=float(finals.max()), p99=float(np.percentile(finals, 99)),
                                           median=float(np.median(finals)), min=float(finals.min()))),
        rate_handling="omega starts at 0 and PERSISTS across outer steps (the real case)",
        parameter_provenance="3U CubeSat per the CubeSat Design Specification (10x10x34 cm, 4.0 kg); inertia is a STATED uniform-rectangular-prism assumption (derives 0.0419/0.0067, simulated 0.049/0.008, and the inertia stress arm spans 0.35x-2.8x so it contains the derivation at 0.85x); reaction-wheel torque 1 mN.m CONFIRMED 2026-08-24 against RL-RW-0.01-Data-Sheet.pdf in the repo ('TORQUE +/-1 mNm at 0.01 Nms'). The earlier PLACEHOLDER blocker is discharged: 0.01 Nms is the wheel's stored MOMENTUM, not its torque.")
    (HERE / "results").mkdir(exist_ok=True)
    suffix = "" if a.band is None else f"_band{a.band[0]}-{a.band[1]}"
    json.dump(res, open(HERE / f"results/sc_deploy_final{suffix}.json", "w"), indent=2)
    if a.trace:
        np.savez_compressed(a.trace, plant="CubeSat", rho_sound=rho,
                            dt_outer=a.dt_outer, **TR)
        print(f"  wrote traces {a.trace}")

    print(f"  seed {a.seed}: {a.episodes} episodes x {a.steps} outer steps ({a.dt_outer}s)")
    print(f"  CONTROL (kinematic) : exits {c_exit}/{a.episodes}  core {c_core}  "
          f"worst dV {c_worst_rise:+.3e}  -> {'PASS' if control_ok else 'FAIL (harness broken)'}")
    if not control_ok:
        raise SystemExit("CONTROL FAILED: kinematic rollouts left D_rho. Harness is broken; the plant "
                         "numbers above are not interpretable and must not be reported.")
    pm = res["plant_mujoco"]
    print(f"  PLANT (MuJoCo)      : exits {p_exit}/{a.episodes} ({100*pm['exit_fraction']:.1f}%)  "
          f"reached core {p_core} ({100*pm['core_fraction']:.1f}%)")
    print(f"                        V rose on {rises}/{total} steps ({100*pm['V_rise_fraction']:.1f}%), "
          f"worst dV {p_worst_rise:+.3e}")
    print(f"                        final theta: {pm['final_theta']}")


if __name__ == "__main__":
    main()
