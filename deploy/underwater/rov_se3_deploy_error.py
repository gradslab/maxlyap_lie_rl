"""The PPO SE(3) policy with an actuator error and its estimated region on the BlueROV2 Heavy in MuJoCo."""
from __future__ import annotations
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jp
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
LIE = HERE.parents[1] / "lie_rl"
sys.path.insert(0, str(LIE)); sys.path.insert(0, str(HERE))
from lie_rl import certificate as C, env as E
from lie_rl.ppo import actor_mean
import hydro

V_RISE_FLOOR = 1e-9
EXEC_CMD_FLOOR = 0.01


def setup(seed, variant="base", which="nominal"):
    ep = E.EnvParams(group="SE3"); G = ep.G; amax = ep.alpha_max
    agent = pickle.load(open(LIE / "results/ppo_SE3_ls-1.5" / f"agent_seed{seed}.pkl", "rb"))
    af = [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in agent.actor]
    bias = actor_mean(af, E.obs(G, G.identity(1)))[0]
    if which == "nominal":
        fid = f"SE3__ppo_SE3_ls-1.5__s{seed}__base"
        cpath, epath = LIE / "results/final/cert" / f"{fid}.pkl", LIE / "results/final/eval" / f"{fid}.json"
    else:
        fid = f"SE3__ppo_SE3_ls-1.5__s{seed}__{variant}"
        for sub in ("results/final", "results/review/pert36", "results/review/pert_r2"):
            tag = "" if sub == "results/final" else "__final"
            cpath, epath = LIE / sub / "cert" / f"{fid}{tag}.pkl", LIE / sub / "eval" / f"{fid}{tag}.json"
            if epath.exists():
                cpath = Path(json.load(open(epath))["cert"]); cpath = cpath if cpath.is_absolute() else LIE / cpath
            if cpath.exists() and epath.exists(): break
        else:
            raise FileNotFoundError(f"no recomputed certificate for {fid}")
    cert = pickle.load(open(cpath, "rb"))
    from lie_rl import anchor as AN
    assert AN.enable_from_cert(cert)
    rec = dict(rho=json.load(open(epath))["fns"]["V"]["level"], cert=str(cpath), eval=str(epath))
    if os.environ.get("ROV_LEVEL"): rec["rho"] = float(os.environ["ROV_LEVEL"])
    cfg = C.CertConfig(**cert["cfg"])
    p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                     alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
    ell = C.make_ell(G)
    V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
    tau_cmd = jax.jit(lambda X: E.clip_tau(G, actor_mean(af, E.obs(G, X)) - bias, amax))
    from lie_rl import variants as Vr
    spec = Vr.parse_variant(variant); M = Vr.mis_matrix("SE3", spec["mis_deg"])
    gain = spec["gain"] if spec.get("axis_seed") is None else np.random.default_rng(1000 + spec["axis_seed"]).uniform(0.7, 1.3, 6)

    def tau_applied(X):
        t = np.asarray(tau_cmd(X))
        if variant == "base": return t
        if M is not None: t = t @ M.T
        t = gain * t; n = np.linalg.norm(t, axis=-1, keepdims=True)
        return t * np.minimum(1.0, amax / np.clip(n, 1e-12, None))
    return G, amax, cfg, V, tau_applied, rec


def X_of(d):
    """lie_rl SE(3) array (qx, qy, qz, qw, tx, ty, tz) of the MuJoCo state."""
    w, x, y, z = d.qpos[3:7]
    return np.array([[x, y, z, w, *d.qpos[:3]]])


def set_state(d, X):
    X = np.asarray(X)[0]
    d.qpos[:3] = X[4:7]; d.qpos[3:7] = [X[3], X[0], X[1], X[2]]; d.qvel[:] = 0


def pose(X):
    X = np.asarray(X)[0]
    return Rot.from_quat(X[:4]).as_matrix(), X[4:7].copy()


def run_plant(G, m, d, inner, X_des, nsub):
    R_des, p_des = pose(X_des)
    for _ in range(nsub):
        d.xfrc_applied[1, :] = 0
        hydro.apply_hydro(m, d)
        d.ctrl[:] = inner(d, R_des, p_des)
        hydro.apply_added_mass(m, d)
        mujoco.mj_step(m, d)


def chart_norm(G, X):
    return float(np.linalg.norm(np.asarray(G.Log(jp.asarray(X)))))


def sample_starts(V, rho, gamma, n, rng):
    out = []
    while len(out) < n:
        z = rng.normal(size=(20000, 6)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        z *= gamma * rng.uniform(size=(20000, 1)) ** (1 / 6)
        keep = np.asarray(V(jp.asarray(z))) <= rho
        out.extend(z[keep][: n - len(out)])
    return np.array(out)


def sweep(a):
    G, amax, cfg, V, tau_of, rec = setup(a.seed)
    rng = np.random.default_rng(1)
    Z = sample_starts(V, rec["rho"], cfg.radius, 60, rng)
    print(f"alpha_max = {amax:.3f}; commanded increments of full length alpha_max from 60 starts in Omega_rho")
    for kp_pos, kd_pos in [(200.0, 120.0), (400.0, 160.0), (800.0, 220.0)]:
        for dt in (0.5, 1.0, 2.0):
            m, d = hydro.load(); inner = hydro.PoseInnerLoop(m, kp_pos=kp_pos, kd_pos=kd_pos)
            nsub = int(round(dt / m.opt.timestep)); ex = []
            for z in Z:
                mujoco.mj_resetData(m, d); X0 = np.asarray(G.Exp(jp.asarray(z[None]))); set_state(d, X0); mujoco.mj_forward(m, d)
                tau = rng.normal(size=6); tau *= amax / np.linalg.norm(tau)
                Xd = np.asarray(G.compose(jp.asarray(X0), G.Exp(jp.asarray(tau[None]))))
                run_plant(G, m, d, inner, Xd, nsub)
                X1 = X_of(d)
                ach = chart_norm(G, np.asarray(G.compose(G.inverse(jp.asarray(X0)), jp.asarray(X1))))
                ex.append(ach / amax)
            ex = np.array(ex)
            print(f"  kp_pos {kp_pos:5.0f} kd_pos {kd_pos:5.0f} dt {dt:.1f} s: executed median {np.median(ex):.3f} min {ex.min():.3f} "
                  f"| saturated substeps {100 * inner.sat / max(inner.calls, 1):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt-outer", type=float, default=1.0)
    ap.add_argument("--kp-pos", type=float, default=400.0)
    ap.add_argument("--kd-pos", type=float, default=160.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--variant", default="m10")
    ap.add_argument("--cert", choices=("recomputed", "nominal"), default="recomputed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sweep:
        return sweep(a)
    G, amax, cfg, V, tau_of, rec = setup(a.seed, a.variant, a.cert); rho = rec["rho"]
    rng = np.random.default_rng(a.seed)
    Z0 = sample_starts(V, rho, cfg.radius, a.episodes, rng)
    Vof = lambda X: float(np.asarray(V(G.Log(jp.asarray(X))))[0])
    TR = {k: np.full((a.episodes, a.steps + 1), np.nan) for k in ("c_cost", "c_V", "p_cost", "p_V", "p_rot", "p_trans")}
    res = {}
    for arm in ("control", "plant"):
        exits = rises = rises_real = 0; execs = []; final = []; worst_rise = 0.0
        if arm == "plant":
            m, d = hydro.load(); inner = hydro.PoseInnerLoop(m, kp_pos=a.kp_pos, kd_pos=a.kd_pos)
            nsub = int(round(a.dt_outer / m.opt.timestep))
        for i, z in enumerate(Z0):
            X = np.asarray(G.Exp(jp.asarray(z[None])))
            if arm == "plant":
                mujoco.mj_resetData(m, d); set_state(d, X); mujoco.mj_forward(m, d)
            Vp = Vof(X); pre = "c_" if arm == "control" else "p_"
            TR[pre + "cost"][i, 0] = chart_norm(G, X); TR[pre + "V"][i, 0] = Vp
            exited = False
            for k in range(a.steps):
                tau = np.asarray(tau_of(jp.asarray(X)))
                Xd = np.asarray(G.compose(jp.asarray(X), G.Exp(jp.asarray(tau))))
                if arm == "control":
                    X1 = Xd
                else:
                    run_plant(G, m, d, inner, Xd, nsub); X1 = X_of(d)
                    cmd = float(np.linalg.norm(tau))
                    if cmd > EXEC_CMD_FLOOR * amax:
                        execs.append(chart_norm(G, np.asarray(G.compose(G.inverse(jp.asarray(X)), jp.asarray(X1)))) / cmd)
                X = X1; Vn = Vof(X)
                if Vn > rho or chart_norm(G, X) > cfg.radius: exited = True
                if Vn - Vp > 1e-12:
                    rises += 1; rises_real += int(Vn - Vp > V_RISE_FLOOR); worst_rise = max(worst_rise, Vn - Vp)
                TR[pre + "cost"][i, k + 1] = chart_norm(G, X); TR[pre + "V"][i, k + 1] = Vn
                if arm == "plant":
                    th, tr = G.err_parts(jp.asarray(X)); TR["p_rot"][i, k + 1] = float(th[0]); TR["p_trans"][i, k + 1] = float(tr[0])
                Vp = Vn
            exits += int(exited); final.append(chart_norm(G, X))
        res[arm] = dict(exits=exits, rises=rises, rises_above_floor=rises_real, worst_rise=worst_rise,
                        final_cost_median=float(np.median(final)),
                        executed_median=(float(np.median(execs)) if execs else None),
                        executed_p5=(float(np.percentile(execs, 5)) if execs else None))
        if arm == "plant":
            res[arm]["saturated_substep_fraction"] = inner.sat / max(inner.calls, 1)
    out = dict(seed=a.seed, variant=a.variant, which_cert=a.cert, cert=rec["cert"], eval=rec["eval"], rho=rho, gamma=cfg.radius, alpha_max=amax, episodes=a.episodes, steps=a.steps,
               dt_outer=a.dt_outer, kp_pos=a.kp_pos, kd_pos=a.kd_pos, **res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
    if a.trace:
        Path(a.trace).parent.mkdir(parents=True, exist_ok=True); np.savez(a.trace, **TR)


if __name__ == "__main__":
    main()
