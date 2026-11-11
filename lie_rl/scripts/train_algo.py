"""Train any of the four learners on any of the four groups.

Usage: python scripts/train_algo.py ALGO GROUP [seeds] [logstd_init]
where ALGO is ppo, trpo, sac or td3.
Writes results/<algo>_<group>_<tag>/.
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import jax
import jax.numpy as jp
import numpy as np

from lie_rl import env as E
from lie_rl.algos import SACConfig, TD3Config, TRPOConfig, train_sac, train_td3, train_trpo
from lie_rl.ppo import PPOConfig, eval_policy, train_ppo


def main(algo="ppo", group="SO2", n_seeds=3, logstd_init=-1.5, steps=None):
    ep = E.EnvParams(group=group)
    G = ep.G
    tag = "" if logstd_init == 0.0 else f"_ls{logstd_init}"
    out = Path(__file__).parents[1] / "results" / f"{algo}_{group}{tag}"
    out.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(999)
    E0 = G.sample(key, 4096, ep.T_max)
    _, rg = E.rollout(G, lambda X: E.geodesic_action(G, X, ep.alpha_max), E0, ep)
    ceiling = float(jp.mean(rg))

    if algo == "ppo":
        cfg = PPOConfig(n_total_steps=steps or 400_000, logstd_init=logstd_init)
        runner = train_ppo
    elif algo == "trpo":
        cfg = TRPOConfig(n_total_steps=steps or 400_000, logstd_init=logstd_init)
        runner = train_trpo
    elif algo == "sac":
        cfg = SACConfig(n_steps=steps or 128_000)
        runner = train_sac
    elif algo == "td3":
        cfg = TD3Config(n_steps=steps or 128_000)
        runner = train_td3
    else:
        raise SystemExit(f"unknown algo {algo}")

    budget = getattr(cfg, "n_total_steps", None) or cfg.n_steps
    print(f"=== {algo.upper()} on {group}: m={G.m}, obs={G.obs_dim}, "
          f"budget={budget}, ceiling={ceiling:.3f} ===", flush=True)

    summary = []
    for seed in range(n_seeds):
        t0 = time.time()
        agent, hist = runner(seed, cfg, ep, verbose=False)
        ret, th, tn = eval_policy(agent, jax.random.key(1000 + seed), 4096, ep)
        rec = dict(algo=algo, group=group, seed=seed, budget=int(budget),
                   geodesic_ceiling=ceiling,
                   eval_return_mean=float(ret.mean()), eval_return_std=float(ret.std()),
                   final_angle_median=float(np.median(th)),
                   final_trans_median=float(np.median(tn)),
                   wall_s=time.time() - t0)
        print(f"  seed {seed}: return {rec['eval_return_mean']:.3f} "
              f"(ceiling {ceiling:.3f}), angle {rec['final_angle_median']:.4f}, "
              f"|t| {rec['final_trans_median']:.4f}, {rec['wall_s']:.0f}s", flush=True)
        summary.append(rec)
        with open(out / f"agent_seed{seed}.pkl", "wb") as f:
            pickle.dump(jax.tree.map(np.asarray, agent), f)
        np.savez(out / f"hist_seed{seed}.npz", hist=np.asarray(hist),
                 budget=budget, ceiling=ceiling)
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    print(f"wrote {out}/summary.json", flush=True)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    g = sys.argv[2] if len(sys.argv) > 2 else "SO2"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ls = float(sys.argv[4]) if len(sys.argv) > 4 else -1.5
    st = int(sys.argv[5]) if len(sys.argv) > 5 else None
    main(a, g, n, ls, st)
