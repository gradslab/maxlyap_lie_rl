"""Table of the trained policies that fail the stability test spr(A) <= 0.99."""
import json

from paths import OUT, RES

GROUPS = [("SO2", r"$\SO(2)$"), ("SO3", r"$\SO(3)$"), ("SE2", r"$\SE(2)$"), ("SE3", r"$\SE(3)$")]
ALGOS = [("ppo", "PPO"), ("trpo", "TRPO"), ("sac", "SAC"), ("td3", "TD3")]
rows = []
for g, gl in GROUPS:
    for a, al in ALGOS:
        for s in range(3):
            f = RES / "stability" / f"{g}__{a}_{g}_ls-1.5__s{s}.json"
            if not f.exists():
                continue
            rho = json.load(open(f))["spr"]
            if rho > 0.99:
                rows.append(f"{gl} & {al} & {s} & {rho:.4g} \\\\")
lines = [r"\begin{tabular}{@{}lllr@{}}", r"\toprule", r"group & algorithm & seed & $\spr(A)$ \\", r"\midrule", *rows,
         r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_fail.tex").write_text("\n".join(lines) + "\n")
print(len(rows), "policies fail the test")
