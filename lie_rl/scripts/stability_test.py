"""Stability test of every trained policy: spectral radius of the Jacobian of the corrected closed loop at the goal."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import jax.numpy as jp
import numpy as np

from lie_rl import env as E
from lie_rl import roa as R
from lie_rl.ppo import actor_mean

RES = Path(__file__).parents[1] / "results"
OUT = RES / "stability"
OUT.mkdir(exist_ok=True)


def test(group, d, seed, h=1e-6):
    ep = E.EnvParams(group=group)
    G, amax, m = ep.G, ep.alpha_max, ep.G.m
    agent = pickle.load(open(RES / d / f"agent_seed{seed}.pkl", "rb"))
    af = [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in agent.actor]
    bias = actor_mean(af, E.obs(G, G.identity(1)))[0]
    T = R.chart_map(af, bias, G, amax, actor_mean)
    J = np.zeros((m, m))
    for i in range(m):
        e = np.zeros(m)
        e[i] = h
        J[:, i] = (np.asarray(T(jp.asarray(e[None]))[0]) - np.asarray(T(jp.asarray(-e[None]))[0])) / (2 * h)
    return dict(group=group, dir=d, seed=seed, spr=float(np.max(np.abs(np.linalg.eigvals(J)))),
                offset_norm=float(jp.linalg.norm(bias)), passes=bool(np.max(np.abs(np.linalg.eigvals(J))) <= 0.99))


if __name__ == "__main__":
    for g in ("SO2", "SO3", "SE2", "SE3"):
        for a in ("ppo", "trpo", "sac", "td3"):
            for s in range(3):
                r = test(g, f"{a}_{g}_ls-1.5", s)
                json.dump(r, open(OUT / f"{g}__{a}_{g}_ls-1.5__s{s}.json", "w"), indent=1)
                print(g, a, s, round(r["spr"], 4), "pass" if r["passes"] else "FAIL")
