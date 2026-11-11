"""Learning curves: mean undiscounted episode return against environment steps, PPO, SAC
and TD3 on the four groups, three seeds (thin) and their mean (thick), against the reference return of u_star.
Style and palette are those of scripts/l4dc_figs.py.

Usage: python scripts/l4dc_curves.py OUTDIR
"""
import glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import l4dc_figs as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = Path(__file__).parents[1] / "results"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "l4dc"
COLS = {"ppo": "#2a78d6", "trpo": "#184f95", "sac": "#eb6834", "td3": "#1baf7a"}
LS = {"ppo": "-", "trpo": (0, (4, 1.5)), "sac": "-", "td3": "-"}
F.style()
fig, ax = plt.subplots(1, 4, figsize=(6.3, 1.75), sharey=True, gridspec_kw=dict(wspace=0.12))
for a, g in zip(ax, F.GROUPS):
    ceil = None
    for algo in ("ppo", "trpo", "sac", "td3"):
        H = []
        for f in sorted(glob.glob(str(RES / f"{algo}_{g}_ls-1.5" / "hist_seed*.npz"))):
            z = np.load(f); H.append(z["hist"]); ceil = float(z["ceiling"]); budget = float(z["budget"])
        if not H: continue
        n = min(len(h) for h in H); H = np.stack([h[:n] for h in H])
        x = (np.arange(n) + 1) * (budget / n) / 1000.0
        if n > 400:
            w = max(1, int(round(n / 40))); ker = np.ones(w) / w
            H = np.stack([np.convolve(h, ker, mode="valid") for h in H]); x = x[w - 1:]
        for h in H: a.plot(x, h, color=COLS[algo], lw=0.4, alpha=0.3, ls=LS[algo])
        a.plot(x, H.mean(0), color=COLS[algo], lw=1.3, ls=LS[algo])
    if ceil is not None: a.axhline(ceil, color=F.INK2, lw=0.8, ls=(0, (3, 2)))
    a.set_title(F.GLAB[g]); a.set_xlabel(r"steps ($10^3$)", labelpad=1)
    a.set_yscale("symlog", linthresh=2.0); a.set_yticks([-2, -5, -10, -20, -40]); a.set_yticklabels(["$-2$", "$-5$", "$-10$", "$-20$", "$-40$"])
    a.minorticks_off(); a.set_ylim(-45, -1); a.set_xlim(0, 400); a.set_xticks([0, 200, 400])
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.grid(axis="y", color=F.GRID, lw=0.4)
ax[0].set_ylabel("episode return", labelpad=1)
fig.legend([Line2D([], [], color=COLS[k], lw=1.3, ls=LS[k]) for k in ("ppo", "trpo", "sac", "td3")] + [Line2D([], [], color=F.INK2, lw=0.8, ls=(0, (3, 2)))],
           ["PPO", "TRPO", "SAC", "TD3", r"reference $u_\star$"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.savefig(OUT / "fig_curves.pdf", dpi=300); fig.savefig(OUT / "fig_curves.png", dpi=200); print("wrote curves")
