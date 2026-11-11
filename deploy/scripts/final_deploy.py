"""Deployment table and figure: the PPO SO(3) policy on the CubeSat and the PPO SE(3)
policy on the BlueROV2 Heavy, both seed 0 with their final certified levels. Reads only the artifacts that
the two studies write (summary JSON and --trace npz); nothing is simulated here.

Usage: python scripts/l4dc_v2_deploy.py OUTDIR      (writes OUTDIR/tab_deploy.tex and OUTDIR/figs/fig_deploy.pdf)
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lie_rl" / "scripts"))
import l4dc_figs as F

out = Path(sys.argv[1])
sc = json.load(open(ROOT / "spacecraft/results/sc_deploy_final.json"))
rov = json.load(open(ROOT / "underwater/results/rov_se3_final.json"))
rho_sc = json.load(open(ROOT.parent / "lie_rl/results/final/eval/SO3__ppo_SO3_ls-1.5__s0__base.json"))["fns"]["V"]["level"]
T_sc = np.load(ROOT / "results/paper/traces/cubesat_final.npz", allow_pickle=True)
T_rov = np.load(ROOT / "results/paper/traces/rov_se3_final.npz", allow_pickle=True)


def sci(x):
    if x == 0: return "0"
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"


pm = sc["plant_mujoco"]; pr = rov["plant"]
rise_sc = float(np.diff(T_sc["p_V"], axis=1).max()); rise_rov = float(np.diff(T_rov["p_V"], axis=1).max())
rows = [
    f"CubeSat & $\\SO(3)$ & {sc.get('dt_outer', 15.7):.1f} s & {sc['control_kinematic']['exits']}/{sc['episodes']} & {pm['exits']}/{sc['episodes']} & "
    f"{sci(rise_sc)} & {sc['plant']['executed_fraction']['median']:.3f} & {sci(pm['final_theta']['median'])} \\\\",
    f"BlueROV2 & $\\SE(3)$ & {rov['dt_outer']:.1f} s & {rov['control']['exits']}/{rov['episodes']} & {pr['exits']}/{rov['episodes']} & "
    f"{sci(rise_rov)} & {pr['executed_median']:.3f} & {sci(pr['final_cost_median'])} \\\\"]
(out / "tab_deploy.tex").write_text("\n".join([r"\begin{tabular}{llrrrrrr}", r"\toprule",
    r"vehicle & group & period & exits ($\Phi_u$) & exits (vehicle) & largest increase of $V$ & carried out & final error \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

import matplotlib.pyplot as plt
F.style()
PLANT_C = F.BLUE; MAP_C = F.MUTED


def band(ax, y, color, ls="-", lw=1.1, z=3):
    t = np.arange(y.shape[1])
    q = [np.nanpercentile(y, s, axis=0) for s in (5, 25, 50, 75, 95)]
    ax.fill_between(t, q[0], q[4], color=color, alpha=0.12, lw=0, zorder=z - 1)
    ax.fill_between(t, q[1], q[3], color=color, alpha=0.25, lw=0, zorder=z - 1)
    ax.plot(t, q[2], color=color, lw=lw, ls=ls, zorder=z)


cols = [("CubeSat, $\\mathrm{SO}(3)$", T_sc["c_th"], T_sc["p_th"], T_sc["c_V"] / rho_sc, T_sc["p_V"] / rho_sc),
        ("BlueROV2, $\\mathrm{SE}(3)$", T_rov["c_cost"], T_rov["p_cost"], T_rov["c_V"] / rov["rho"], T_rov["p_V"] / rov["rho"])]
fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.3), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.28))
for j, (name, c_cost, p_cost, c_v, p_v) in enumerate(cols):
    a = axs[0, j]
    band(a, np.maximum(c_cost, 1e-17), MAP_C, ls=(0, (3, 2)), lw=0.9)
    band(a, np.maximum(p_cost, 1e-17), PLANT_C)
    a.axhline(rov["gamma"], color=F.INK2, lw=0.5, ls=":")
    a.set_yscale("log"); a.set_ylim(1e-17, 8); a.set_title(name)
    from matplotlib.ticker import LogLocator, NullLocator
    a.yaxis.set_major_locator(LogLocator(numticks=5)); a.yaxis.set_minor_locator(NullLocator())
    b = axs[1, j]
    band(b, c_v, MAP_C, ls=(0, (3, 2)), lw=0.9)
    band(b, p_v, PLANT_C)
    b.axhline(1.0, color=F.ORANGE, lw=1.1)
    b.set_ylim(-0.03, 1.12); b.set_xlabel("outer step", labelpad=1)
    for x in (a, b):
        for s in ("top", "right"): x.spines[s].set_visible(False)
axs[0, 0].set_ylabel(r"$\|\mathrm{Log}\,E\|$", labelpad=1); axs[1, 0].set_ylabel(r"$V/\rho$", labelpad=1)
axs[0, 0].text(1, 3.2, r"$\gamma$", color=F.INK2, fontsize=6)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=PLANT_C, lw=1.1), Line2D([], [], color=MAP_C, lw=0.9, ls=(0, (3, 2))),
            Line2D([], [], color=F.ORANGE, lw=1.1)], ["plant (MuJoCo)", "kinematic map", r"level $\rho$"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
(out / "figs").mkdir(exist_ok=True)
fig.savefig(out / "figs/fig_deploy.pdf", dpi=300); fig.savefig(out / "figs/fig_deploy.png", dpi=200)
print("\n".join(rows))
print("sc: control exits", sc["control_kinematic"]["exits"], "plant exits", pm["exits"], "worst rise", pm["worst_V_rise"], "rho", rho_sc)
print("rov:", json.dumps({k: rov[k] for k in ("rho", "control", "plant")}))
