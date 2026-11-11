"""BlueROV2 deployment under conditions the kinematic model does not assume.

Usage: python underwater/rov_F_deploy.py OUT_JSON [--seed 0] [--episodes 200] [--steps 40] [--dt-outer 2.0]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rov_se3_deploy_error as RE
import jax.numpy as jp
import mujoco
import hydro


def run_plant_track(m, d, inner, X_des, nsub):
    R_des, p_des = RE.pose(X_des); vmax = wmax = 0.0
    for _ in range(nsub):
        d.xfrc_applied[1, :] = 0
        hydro.apply_hydro(m, d); d.ctrl[:] = inner(d, R_des, p_des); hydro.apply_added_mass(m, d); mujoco.mj_step(m, d)
        vmax = max(vmax, float(np.linalg.norm(d.qvel[:3]))); wmax = max(wmax, float(np.linalg.norm(d.qvel[3:6])))
    return vmax, wmax


def ball(rng, r):
    u = rng.normal(size=3); return u * r * rng.uniform() ** (1 / 3) / np.linalg.norm(u)


def run_arm(G, Z0, V, tau_of, rho, gamma, amax, dt, steps, kp, kd, v0, w0, rng):
    m, d = hydro.load(); inner = hydro.PoseInnerLoop(m, kp_pos=kp, kd_pos=kd); nsub = int(round(dt / m.opt.timestep))
    Vof = lambda X: float(np.asarray(V(G.Log(jp.asarray(X))))[0])
    exits = conv = 0; rise = -np.inf; execs = []; finals = []; vmax = wmax = 0.0
    for z in Z0:
        X = np.asarray(G.Exp(jp.asarray(z[None]))); mujoco.mj_resetData(m, d); RE.set_state(d, X)
        if v0 > 0 or w0 > 0: d.qvel[:3] = ball(rng, v0); d.qvel[3:6] = ball(rng, w0)
        mujoco.mj_forward(m, d); Vp = Vof(X); exited = False
        for _ in range(steps):
            tau = np.asarray(tau_of(jp.asarray(X))); Xd = np.asarray(G.compose(jp.asarray(X), G.Exp(jp.asarray(tau))))
            vm, wm = run_plant_track(m, d, inner, Xd, nsub); vmax = max(vmax, vm); wmax = max(wmax, wm)
            X1 = RE.X_of(d); cmd = float(np.linalg.norm(tau))
            if cmd > RE.EXEC_CMD_FLOOR * amax:
                execs.append(RE.chart_norm(G, np.asarray(G.compose(G.inverse(jp.asarray(X)), jp.asarray(X1)))) / cmd)
            X = X1; Vn = Vof(X)
            exited |= (Vn > rho) or (RE.chart_norm(G, X) > gamma); rise = max(rise, Vn - Vp); Vp = Vn
        exits += int(exited); th, tr = G.err_parts(jp.asarray(X)); conv += int(float(th[0]) < 0.05 and float(tr[0]) < 0.05)
        finals.append(RE.chart_norm(G, X))
    return dict(exits=exits, converged=conv, episodes=len(Z0), largest_V_increase=float(rise), v_max=vmax, w_max=wmax,
                carried_out_median=float(np.median(execs)) if execs else None, carried_out_p5=float(np.percentile(execs, 5)) if execs else None,
                saturated_substep_fraction=inner.sat / max(inner.calls, 1), final_error_median=float(np.median(finals)), final_error_max=float(np.max(finals)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--dt-outer", type=float, default=2.0)
    ap.add_argument("--kp-pos", type=float, default=400.0); ap.add_argument("--kd-pos", type=float, default=160.0)
    a = ap.parse_args(); t0 = time.time()
    G, amax, cfg, V, tau_of, rec = RE.setup(a.seed, "base", "nominal"); rho = rec["rho"]
    Z0 = RE.sample_starts(V, rho, cfg.radius, a.episodes, np.random.default_rng(a.seed))
    args = (G, Z0, V, tau_of, rho, cfg.radius, amax)
    res = {"nominal": run_arm(*args, a.dt_outer, a.steps, a.kp_pos, a.kd_pos, 0.0, 0.0, np.random.default_rng(1))}
    vm, wm = res["nominal"]["v_max"], res["nominal"]["w_max"]; print("nominal", json.dumps(res["nominal"]), flush=True)
    for name, frac, dt, ns in (("rate25", 0.25, a.dt_outer, a.steps), ("rate50", 0.5, a.dt_outer, a.steps), ("half", 0.0, a.dt_outer / 2, 2 * a.steps)):
        res[name] = dict(run_arm(*args, dt, ns, a.kp_pos, a.kd_pos, frac * vm, frac * wm, np.random.default_rng(2)),
                         initial_v_radius=frac * vm, initial_w_radius=frac * wm, dt_outer=dt, steps=ns)
        print(name, json.dumps(res[name]), flush=True)
    out = dict(vehicle="BlueROV2", seed=a.seed, cert=rec["cert"], eval=rec["eval"], rho=rho, episodes=a.episodes, steps=a.steps,
               dt_outer=a.dt_outer, kp_pos=a.kp_pos, kd_pos=a.kd_pos, arms=res, wall_s=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
