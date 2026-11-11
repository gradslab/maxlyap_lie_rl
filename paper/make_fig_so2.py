"""Appendix figure: the region of SO(2) TRPO s2 estimated by the level rule and the region proved by interval arithmetic."""
import sys, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
C_CAND = "#2d7fd3"; C_PROVED = "#1a1a1a"; C_FAIL = "#c81d3a"; C_EXCL = "#d3d1ce"; INK, INK2, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})
d = np.load(sys.argv[1]); out = sys.argv[2]
z, V, rho, lvl, gam, delta = d["z"], d["V"], float(d["rho"]), float(d["lvl"]), float(d["gamma"]), float(d["delta"])
LO, HI, gU, Vlo, l2hi, r_b = d["LO"], d["HI"], d["gU"], d["Vlo"], d["l2hi"], float(d["r_b"])
fail = (LO > 2.7488921) & (HI <= gam + 1e-12)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.1), gridspec_kw=dict(wspace=0.32))
ax.plot(z, V, color=C_CAND, lw=0.9)
ax.axhline(rho, color=MUTED, lw=0.6, ls="--")
ax.text(-gam + 0.05, rho + 0.8, r"level $\rho$", color=INK2, fontsize=7)
def bar(y, mask, color, label):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2] - 1):
        ax.plot([z[a], z[b]], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(-gam - 0.08, y, label, ha="right", va="center", fontsize=7, color=INK2)
bar(-4.0, V <= rho, C_CAND, "estimated")
bar(-7.5, V <= lvl, C_PROVED, "proved")
ax.set_xlim(-gam - 0.05, gam + 0.05); ax.set_ylim(-10, 40)
ax.set_xlabel(r"rotation error $z$ (rad)"); ax.set_ylabel(r"$V(z)$")
ax.set_xticks([-gam, -1.5, 0, 1.5, gam]); ax.set_xticklabels([r"$-\gamma$", "$-1.5$", "0", "1.5", r"$\gamma$"])
ax.spines[["top", "right"]].set_visible(False); ax.set_title("(a)", loc="left")
ax.plot([gam], [-4.0], marker="v", ms=4, color=C_FAIL, clip_on=False, zorder=5)
ax.annotate("containment fails on\none cell at the edge", xy=(gam, -3.2), xytext=(0.75, 22), fontsize=6.5, color=C_FAIL,
            ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=C_FAIL))
m = (Vlo <= lvl) & (l2hi >= r_b ** 2)
c = 0.5 * (LO + HI)
bx.semilogy(c[m], -gU[m], ".", ms=0.8, color=C_PROVED, label="proved bound", rasterized=True)
o = (np.abs(z) >= delta) & (V <= lvl)
bx.semilogy(z[o][::20], -d["dV"][o][::20], "-", lw=0.6, color=C_CAND, label="value at points")
bx.axvspan(-delta, delta, color=C_EXCL, lw=0)
bx.text(0.12, 4.2e-3, r"$N_b$", ha="center", fontsize=7, color=INK2)
bx.set_xlim(-gam - 0.05, gam + 0.05); bx.set_ylim(3e-3, 20)
bx.set_xticks([-gam, -1.5, 0, 1.5, gam]); bx.set_xticklabels([r"$-\gamma$", "$-1.5$", "0", "1.5", r"$\gamma$"])
bx.set_xlabel(r"rotation error $z$ (rad)"); bx.set_ylabel(r"decrease $-\delta V$")
bx.legend(frameon=False, loc="lower left", handlelength=1.2, markerscale=6)
bx.spines[["top", "right"]].set_visible(False); bx.set_title("(b)", loc="left")
fig.savefig(out, bbox_inches="tight", dpi=300); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
