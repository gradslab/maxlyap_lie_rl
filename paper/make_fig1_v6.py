"""Figure 1: panels (a) and (b) copied as vector graphics from paper/assets/fig1_panels_ab.pdf, and panel (c) drawn here."""
import json, sys, statistics as st
from pathlib import Path
import numpy as np, fitz
import matplotlib; matplotlib.use("Agg"); import matplotlib as mpl, matplotlib.pyplot as plt
from paths import RES as R, OUT, REPO
HERE = OUT / "figs"; HERE.mkdir(parents=True, exist_ok=True)
OLD = REPO / "paper/assets/fig1_panels_ab.pdf"; FID = "SE2__ppo_SE2_ls-1.5__s0__m40"
W, H = 6.3, 2.25; FS = 1.3; RECT_C = [0.80, 0.17, 0.19, 0.64]; XCUT = 326.0
C_CAND, C_BASE, C_VD, C_H = "#2d7fd3", "#f2a516", "#6f6f6f", "#009E73"; INK, INK2, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"; BAND, STEM = "#e9eef5", "#c9d9ea"
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})
for k_ in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"): mpl.rcParams[k_] *= FS
Hb = {r["id"]: r for r in json.load(open(R / "review/hinge_base/hb_levels.json"))["comparison"]}
FL = json.load(open(R / "review/F/FINAL_LEVELS.json"))["loops"]
ids = sorted(Hb); orc = np.array([100 * json.load(open(R / f"rx/oracle/{i}.json"))["region_mu0"] for i in ids])
pct = lambda key: np.array([100 * FL[i][key]["region_mu0"] for i in ids]) / orc * 100
v, vd, vp = pct("V"), pct("Vd"), pct("VP"); vh = np.array([100 * Hb[i]["hinge"] for i in ids]) / orc * 100
for i in ids: assert abs(FL[i]["V"]["region_mu0"] - Hb[i]["V"]) < 1e-12
print("medians over oracle: V %.1f, V_h %.1f, V_d %.1f, V_P %.1f" % (np.median(v), np.median(vh), np.median(vd), np.median(vp)))
ub = np.maximum(vd, vp); o_ = np.argsort(ub, kind="stable"); x = np.arange(len(ids)); ex = [k for k, j in enumerate(o_) if ids[j] == FID][0]
fig = plt.figure(figsize=(W, H)); axc = fig.add_axes(RECT_C)
axc.axvspan(ex - 0.5, ex + 0.5, color=BAND, lw=0, zorder=0)
for k, j in enumerate(o_): axc.plot([k, k], [min(vd[j], vp[j]), v[j]], color=STEM, lw=1.6, zorder=1, solid_capstyle="round")
axc.axhline(100, color=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=0.5)
axc.scatter(x, vd[o_], s=9, marker="v", color=C_VD, edgecolor="white", lw=0.3, zorder=2, label="$V_d$")
axc.scatter(x, vp[o_], s=9, color=C_BASE, edgecolor="white", lw=0.3, zorder=2.5, label="$V_P$")
axc.scatter(x, vh[o_], s=9, marker="s", color=C_H, edgecolor="white", lw=0.3, zorder=2.7, label="$V_h$")
axc.scatter(x, v[o_], s=9, color=C_CAND, edgecolor="white", lw=0.3, zorder=3, label="$V$")
axc.annotate("loop of (b)", xy=(ex + 0.3, ub[o_][ex] - 4), xytext=(ex + 2.5, 40), ha="left", va="center", fontsize=7 * FS, color=INK,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0.5, shrinkB=0.5))
axc.set_ylim(0, 104); axc.set_yticks([0, 50, 100]); axc.set_xticks([]); axc.set_xlim(-1, len(ids))
axc.set_ylabel(r"% of oracle region", labelpad=2); axc.set_xlabel("closed loop", labelpad=3)
for s_ in ("top", "right"): axc.spines[s_].set_visible(False)
h_, l_ = axc.get_legend_handles_labels(); order = [3, 2, 0, 1]
leg = axc.legend([h_[k] for k in order], [l_[k] for k in order], loc="lower center", bbox_to_anchor=(0.38, 1.0), ncol=4, frameon=False,
           handlelength=0.8, handletextpad=0.0, columnspacing=0.35, borderaxespad=0.15, markerscale=1.2, fontsize=6.5 * FS)
fig.canvas.draw(); bb = leg.get_window_extent(fig.canvas.get_renderer()); print("legend right edge / figure width: %.3f" % (bb.x1 / fig.bbox.width))
assert bb.x1 < fig.bbox.width - 0.08 * fig.dpi, "legend too wide"
fig.text(RECT_C[0] + RECT_C[2] / 2 - 0.03, 0.955, "(c) All actuator errors", ha="center", va="top", fontsize=8 * FS)
fig.savefig(HERE / "panel_c_v6.pdf", dpi=600); plt.close(fig)
old = fitz.open(OLD); new = fitz.open(HERE / "panel_c_v6.pdf"); pw, ph = old[0].rect.width, old[0].rect.height
assert abs(new[0].rect.width - pw) < 1 and abs(new[0].rect.height - ph) < 1, (new[0].rect, old[0].rect)
out = fitz.open(); pg = out.new_page(width=pw, height=ph)
pg.show_pdf_page(fitz.Rect(0, 0, XCUT, ph), old, 0, clip=fitz.Rect(0, 0, XCUT, ph))
pg.show_pdf_page(fitz.Rect(XCUT, 0, pw, ph), new, 0, clip=fitz.Rect(XCUT, 0, pw, ph))
out.save(HERE / "fig1_actuator.pdf"); (HERE / "panel_c_v6.pdf").unlink(); print("written", HERE / "fig1_actuator.pdf")
