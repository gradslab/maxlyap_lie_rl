"""Shared figure style and palette."""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np

RES = Path(__file__).parents[1] / "results"; L4 = RES / "l4dc"; CACHE = L4 / "figdata"; CACHE.mkdir(exist_ok=True)
GROUPS = ["SO2", "SO3", "SE2", "SE3"]; ALGOS = ["ppo", "trpo", "sac", "td3"]
GLAB = {"SO2": r"$\mathrm{SO}(2)$", "SO3": r"$\mathrm{SO}(3)$", "SE2": r"$\mathrm{SE}(2)$", "SE3": r"$\mathrm{SE}(3)$"}
ALAB = {"ppo": "PPO", "trpo": "TRPO", "sac": "SAC", "td3": "TD3"}
SLICE = {"SO3": (0, 1), "SE2": (0, 1), "SE3": (0, 3)}
AXLAB = {"SO3": (r"$z_1$ (rotation)", r"$z_2$ (rotation)"), "SE2": (r"$z_1$ (rotation)", r"$z_2$ (translation)"),
         "SE3": (r"$z_1$ (rotation)", r"$z_4$ (translation)")}


def tagof(a): return "" if a == "ppo" else f"_{a}"


def data(group, algo, seed):
    import jax, jax.numpy as jp
    from lie_rl import certificate as C, env as E, roa as R
    from lie_rl.ppo import actor_mean
    d = f"{algo}_{group}_ls-1.5"
    rec = json.load(open(L4 / f"roa_{group}{tagof(algo)}_seed{seed}.json"))
    ep = E.EnvParams(group=group); G = ep.G; m = G.m; amax = ep.alpha_max
    agent = pickle.load(open(RES / d / f"agent_seed{seed}.pkl", "rb"))
    af = [(jp.asarray(np.asarray(W, np.float64)), jp.asarray(np.asarray(b, np.float64))) for W, b in agent.actor]
    bias = actor_mean(af, E.obs(G, G.identity(1)))[0]
    T = jax.jit(R.chart_map(af, bias, G, amax, actor_mean))
    cert = pickle.load(open(RES / "cert" / f"{d}_seed{seed}{rec['suffix']}.pkl", "rb"))
    cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius
    p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                     alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
    ell = C.make_ell(G); V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
    vm = jax.jit(C.make_vm(T, ell, cfg._replace(rollout_len=600)))
    policy_step = lambda X: E.apply_action(G, X, actor_mean(af, E.obs(G, X)) - bias, amax)
    out = dict(m=m, gamma=gamma, amax=amax, rho=rec["rho"], r_d=rec["dist_radius"], r_b=cfg.eps_ball)
    if m == 1:
        th = np.linspace(-np.pi, np.pi, 4001); zt = jp.asarray(th)[:, None]
        Vt = np.asarray(V(zt)); Ft = np.asarray(T(zt))[:, 0]
        out.update(th=th, V=Vt, Vm=np.asarray(vm(zt)), F=Ft, dV=np.asarray(V(T(zt))) - Vt,
                   roots=np.asarray(rec["fixed_points"].get("roots", [])))
    else:
        i1, i2 = SLICE[group]; n = 321; g = np.linspace(-np.pi, np.pi, n); Z1, Z2 = np.meshgrid(g, g)
        def lift(a, b):
            zs = np.zeros((a.size, m)); zs[:, i1] = a.ravel(); zs[:, i2] = b.ravel(); return zs
        zs = lift(Z1, Z2); inb = np.linalg.norm(zs, axis=-1) < np.pi - 1e-3
        Vs = np.array(V(jp.asarray(zs))); dVs = np.array(V(T(jp.asarray(zs)))) - Vs
        Vs[~inb] = np.nan; dVs[~inb] = np.nan
        z2 = np.asarray(C.sample_ball(jax.random.key(3), 10, 2, 0.95 * gamma)); zst = lift(z2[:, 0], z2[:, 1])
        Xc = G.Exp(jp.asarray(zst)); traj = [np.asarray(G.Log(Xc))]
        for _ in range(60): Xc = policy_step(Xc); traj.append(np.asarray(G.Log(Xc)))
        traj = np.stack(traj)[:, :, [i1, i2]]
        zf = np.asarray(C.sample_ball(jax.random.key(4), 4000, m, gamma))
        out.update(g=g, V=Vs.reshape(n, n), dV=dVs.reshape(n, n), traj=traj,
                   Vf=np.asarray(V(jp.asarray(zf))), Vmf=np.asarray(vm(jp.asarray(zf))))
    if m == 1:
        zf = np.asarray(C.sample_ball(jax.random.key(4), 4000, m, gamma))
        out.update(Vf=np.asarray(V(jp.asarray(zf))), Vmf=np.asarray(vm(jp.asarray(zf))))
    np.savez(CACHE / f"{group}_{algo}_{seed}.npz", **out)


INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
ORANGE, BLUE = "#eb6834", "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f4b9ae", "#e98476", "#e34948", "#a32322"]


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.labelcolor": INK, "text.color": INK, "axes.titlepad": 3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42})


def cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", SEQ); seq.set_bad("white")
    div = LinearSegmentedColormap.from_list("div", DIV); div.set_bad("white")
    return seq, div


def circles(ax, D):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((0, 0), D["gamma"], fill=False, lw=0.5, color=MUTED))
    if D["r_d"] > 1e-6:
        ax.add_patch(plt.Circle((0, 0), D["r_d"], fill=False, lw=0.8, ls=(0, (3, 2)), color=INK2))


def row(axs, D, group, first_row, last_row, cbar=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    seq, div = cmaps(); rho = float(D["rho"]); m = int(D["m"])
    if m == 1:
        th, V, Vm = D["th"], D["V"], D["Vm"]; gam = float(D["gamma"])
        a = axs[0]
        cert = (np.abs(th) <= gam) & (V <= rho)
        a.fill_between(th, 1e-4, 1e4, where=cert, color="#eaf2fc", lw=0, step="mid")
        a.plot(th, np.where(np.isfinite(Vm), Vm, np.nan), color=SEQ[2], lw=2.2, solid_capstyle="round", label=r"$V_{\mathrm{m}}$")
        a.plot(th, V, color=INK, lw=0.8, label=r"$V$")
        if rho > 0: a.axhline(rho, color=ORANGE, lw=1.0)
        a.set_yscale("log")
        from matplotlib.ticker import LogLocator, NullLocator
        a.yaxis.set_major_locator(LogLocator(numticks=4)); a.yaxis.set_minor_locator(NullLocator())
        top = 3 * max(rho, np.nanmax(V[np.abs(th) <= gam])); a.set_ylim(1e-3, top); gate_note(a, D)
        a.legend(frameon=False, loc="lower center", ncol=2, handlelength=1.2, columnspacing=0.8)
        b = axs[1]; rel = D["dV"] / np.maximum(V, 1e-12)
        rel = np.where(2 * np.sqrt(2) * np.abs(np.sin(th / 2)) < float(D["r_b"]), np.nan, rel)
        b.axhline(0, color=MUTED, lw=0.5)
        b.plot(th, np.clip(rel, -1.05, 0.5), color=BLUE, lw=0.9)
        b.fill_between(th, np.clip(rel, 0, 0.5), 0, color=DIV[-2], lw=0)
        b.set_ylim(-1.05, 0.3)
        c = axs[2]; c.plot(th, th, color=GRID, lw=0.6); c.plot(th, D["F"], color=INK, lw=0.8)
        for r_ in D["roots"]: c.plot(r_, r_, "o", ms=3.2, mfc="white", mec=INK, mew=0.7)
        for a_ in axs:
            a_.axvline(gam, color=MUTED, lw=0.5); a_.axvline(-gam, color=MUTED, lw=0.5)
            a_.set_xlim(-np.pi, np.pi); a_.set_xticks([-np.pi, 0, np.pi]); a_.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
            for s in ("top", "right"): a_.spines[s].set_visible(False)
        if last_row:
            for a_ in axs: a_.set_xlabel(r"$z$", labelpad=1)
        if first_row:
            axs[0].set_title(r"$V$, $V_{\mathrm{m}}$ and the level $\rho$"); axs[1].set_title(r"$(V(F(z)) - V(z))/V(z)$"); axs[2].set_title(r"closed-loop map $F$")
        return
    g = D["g"]; V = D["V"]; rel = D["dV"] / np.where(np.isnan(V), 1, np.maximum(V, 1e-12))
    a = axs[0]
    vmin = max(1e-3, np.nanpercentile(V, 0.5)); vmax = np.nanmax(V)
    im0 = a.pcolormesh(g, g, np.clip(V, vmin, None), norm=LogNorm(vmin, vmax), cmap=seq, shading="auto", rasterized=True)
    gate_note(a, D)
    G1, G2 = np.meshgrid(g, g); Rr = np.hypot(G1, G2); inpi = Rr < np.pi - 1e-3
    M = ((Rr <= float(D["gamma"])) & (np.nan_to_num(V, nan=np.inf) <= rho)).astype(float)
    def region(ax_, veil):
        ax_.contourf(g, g, np.where(inpi, M, np.nan), levels=[-0.5, 0.5], colors="white", alpha=veil)
        if M.any(): ax_.contour(g, g, M, levels=[0.5], colors=ORANGE, linewidths=1.1)
    region(a, 0.55)
    circles(a, D)
    for tr in np.moveaxis(D["traj"], 1, 0):
        a.plot(tr[:, 0], tr[:, 1], color="white", lw=1.6, alpha=0.9, solid_capstyle="round")
        a.plot(tr[:, 0], tr[:, 1], color=INK, lw=0.6, solid_capstyle="round")
        a.plot(tr[0, 0], tr[0, 1], "o", ms=2.2, mfc=INK, mec="white", mew=0.4)
    b = axs[1]
    pos = np.nanmax(rel); hi = max(0.05, min(1.0, pos if np.isfinite(pos) else 0.05))
    im1 = b.pcolormesh(g, g, np.clip(rel, -1, hi), norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=hi), cmap=div, shading="auto", rasterized=True)
    region(b, 0.45)
    circles(b, D)
    i1l, i2l = AXLAB[group]
    for a_ in axs[:2]:
        a_.set_aspect("equal"); a_.set_xlim(-np.pi, np.pi); a_.set_ylim(-np.pi, np.pi)
        a_.set_xticks([-np.pi, 0, np.pi]); a_.set_yticks([-np.pi, 0, np.pi])
        a_.set_xticklabels([r"$-\pi$", "0", r"$\pi$"]); a_.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        a_.set_xlabel(i1l, labelpad=1)
    axs[0].set_ylabel(i2l, labelpad=0)
    if cbar:
        c0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.03); c0.outline.set_linewidth(0.3); c0.ax.tick_params(labelsize=5, width=0.3, length=1.5, which="both"); c0.ax.minorticks_off()
        c1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.03); c1.outline.set_linewidth(0.3); c1.ax.tick_params(labelsize=5, width=0.3, length=1.5)
    c = axs[2]
    Vf, Vmf = D["Vf"], D["Vmf"]; ok = np.isfinite(Vmf) & (Vmf > 0) & (Vf > 0)
    lo = max(1e-3, np.percentile(np.minimum(Vf[ok], Vmf[ok]), 0.5)); up = max(Vf[ok].max(), np.percentile(Vmf[ok], 99.5))
    c.plot([lo, up], [lo, up], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    c.scatter(Vmf[ok], Vf[ok], s=1.2, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    if rho > 0: c.axhline(rho, color=ORANGE, lw=1.0)
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lo, up * 1.3); c.set_ylim(lo, up * 1.3)
    from matplotlib.ticker import LogLocator, NullLocator
    for ax_ in (c.xaxis, c.yaxis): ax_.set_major_locator(LogLocator(numticks=4)); ax_.set_minor_locator(NullLocator())
    c.set_xlabel(r"$V_{\mathrm{m}}$", labelpad=1); c.set_ylabel(r"$V$", labelpad=0); c.set_box_aspect(1)
    for s in ("top", "right"): c.spines[s].set_visible(False)
    if first_row:
        axs[0].set_title(r"$V$ on the slice, level $\rho$"); axs[1].set_title(r"$(V(F(z)) - V(z))/V(z)$"); axs[2].set_title(r"$V$ against $V_{\mathrm{m}}$")


GATE = 0.99


def load(group, algo, seed):
    f = CACHE / f"{group}_{algo}_{seed}.npz"
    if not f.exists(): return None
    D = dict(np.load(f)); rec = json.load(open(L4 / f"roa_{group}{tagof(algo)}_seed{seed}.json"))
    D["spr"] = rec["fixed_points"]["rho_goal"]; D["gated"] = D["spr"] > GATE
    if D["gated"]: D["rho"] = 0.0; D["r_d"] = 0.0
    return D


def gate_note(ax, D):
    if D.get("gated"):
        ax.text(0.03, 0.97, rf"$\mathrm{{spr}}(DF(0)) = {D['spr']:.2f}$: no region", transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, color=INK, bbox=dict(fc="white", ec=GRID, lw=0.4, pad=1.5, alpha=0.9))


def draw(outdir):
    import matplotlib.pyplot as plt
    style(); outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    W = 6.3
    for group in GROUPS:
        for algo in ALGOS:
            Ds = [load(group, algo, s) for s in range(3)]
            if any(D is None for D in Ds): continue
            h = 1.55 if group == "SO2" else 1.95
            fig, axs = plt.subplots(3, 3, figsize=(W, 3 * h), gridspec_kw=dict(wspace=0.42 if group != "SO2" else 0.3, hspace=0.42))
            for s, D in enumerate(Ds):
                row(axs[s], D, group, s == 0, s == 2)
                rec = json.load(open(L4 / f"roa_{group}{tagof(algo)}_seed{s}.json"))
                axs[s, 0].annotate(f"seed {s}", xy=(-0.42 if group != "SO2" else -0.3, 0.5), xycoords="axes fraction", rotation=90,
                                   va="center", ha="center", fontsize=7, color=INK)
            fig.savefig(outdir / f"fig_{group}_{algo}.pdf", dpi=300); plt.close(fig)
    rows = [("SO3", "td3", 1), ("SE3", "ppo", 1)]; Ds = [load(*r) for r in rows]
    if all(D is not None for D in Ds):
        fig = plt.figure(figsize=(W, 2 * 1.8))
        gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.42)
        for i, ((g_, a_, s_), D) in enumerate(zip(rows, Ds)):
            axs = [fig.add_subplot(gs[i, j]) for j in range(3)]
            row(axs, D, g_, i == 0, True)
            axs[0].annotate(f"{GLAB[g_]}, {ALAB[a_]} seed {s_}", xy=(-0.42 if g_ != "SO2" else -0.3, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center", fontsize=7)
        fig.savefig(outdir / "fig_main.pdf", dpi=300); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "data":
        data(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        draw(sys.argv[2])
