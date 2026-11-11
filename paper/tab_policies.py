"""Table of the trained policies: return, final errors, output at the goal and the stability test."""
import json

from paths import OUT, RES

GROUPS = [("SO2", r"$\SO(2)$"), ("SO3", r"$\SO(3)$"), ("SE2", r"$\SE(2)$"), ("SE3", r"$\SE(3)$")]
ALGOS = [("ppo", "PPO"), ("trpo", "TRPO"), ("sac", "SAC"), ("td3", "TD3")]
common = json.load(open(RES / "l4dc" / "policy_eval_common.json"))
rows = []
for k, (g, lab) in enumerate(GROUPS):
    first = True
    for a, alab in ALGOS:
        for s in range(3):
            f = RES / "stability" / f"{g}__{a}_{g}_ls-1.5__s{s}.json"
            if not f.exists():
                continue
            r, ce = json.load(open(f)), common[g][f"{a}_{s}"]
            rows.append(f"{lab if first else ''} & {alab if s == 0 else ''} & {s} & {ce['return_mean']:.3f} & "
                        f"{common[g]['reference']['return_mean']:.3f} & {ce['rot_median']:.4f} & {ce['trans_median']:.4f} & "
                        f"{r['offset_norm']:.4f} & {r['spr']:.3f} \\\\")
            first = False
    if g != "SE3":
        rows.append(r"\midrule")
(OUT / "tab_policies.tex").write_text("\n".join([r"\begin{tabular}{lllrrrrrr}", r"\toprule",
    r"group & algorithm & seed & return & reference & rotation & translation & offset & $\spr(DF(0))$ \\", r"\midrule", *rows,
    r"\bottomrule", r"\end{tabular}"]))
