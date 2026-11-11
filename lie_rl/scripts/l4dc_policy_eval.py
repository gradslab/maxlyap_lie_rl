"""Every trained policy and the reference u_star on one common set of 4096 starts of mu_0 per group:
undiscounted return over 50 steps, median final rotation angle and translation norm. The policy is the
trained mean network before the repair of Section 4.1. Writes results/l4dc/policy_eval_common.json.

Usage: JAX_ENABLE_X64=1 python scripts/l4dc_policy_eval.py
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import jax, jax.numpy as jp, numpy as np
from lie_rl import env as E
from lie_rl.ppo import actor_mean

RES = Path(__file__).parents[1] / "results"; out = {}
for g in ("SO2", "SO3", "SE2", "SE3"):
    ep = E.EnvParams(group=g); G = ep.G
    X0 = G.sample(jax.random.key(2027), 4096, ep.T_max)
    def summ(policy):
        Xf, ret = E.rollout(G, policy, X0, ep); th, tr = G.err_parts(Xf)
        return dict(return_mean=float(ret.mean()), rot_median=float(np.median(th)), trans_median=float(np.median(tr)))
    out[g] = dict(reference=summ(lambda X: E.geodesic_action(G, X, ep.alpha_max)))
    for a in ("ppo", "trpo", "sac", "td3"):
        for s in range(3):
            ag = pickle.load(open(RES / f"{a}_{g}_ls-1.5" / f"agent_seed{s}.pkl", "rb"))
            af = [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in ag.actor]
            out[g][f"{a}_{s}"] = summ(lambda X, af=af: actor_mean(af, E.obs(G, X)))
    print(g, "reference", out[g]["reference"])
json.dump(out, open(RES / "l4dc" / "policy_eval_common.json", "w"), indent=1)
