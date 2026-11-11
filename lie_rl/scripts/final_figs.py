"""Figures of the final method (rx_reports/final_method.md, Part III), drawn with the style of l4dc_figs.py."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import l4dc_figs as F

RES = Path(__file__).parents[1] / "results"; FIN = RES / "final"; CACHE = FIN / "figdata"; CACHE.mkdir(parents=True, exist_ok=True)
ORACLE = RES / "rx" / "oracle"


def data(fid):
    import pickle, jax, jax.numpy as jp
    from lie_rl import certificate as C, anchor as AN, variants as Vr
    G_, d, s, v = fid.split("__"); seed = int(s[1:])
    cl = Vr.ClosedLoop(G_, d, seed, v); G = cl.G; m = cl.m; T = jax.jit(cl.chart_map())
    rec = json.load(open(FIN / "eval" / f"{fid}.json"))
    cert = pickle.load(open(FIN / "cert" / f"{fid}.pkl", "rb")); assert AN.enable_from_cert(cert)
    cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius
    p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                     alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
    ell = C.make_ell(G); V = jax.jit(lambda z: C.V_chart(p, cfg, ell, z))
    orc = json.load(open(ORACLE / f"{fid}.json")) if (ORACLE / f"{fid}.json").exists() else None
    N = int(orc["N_rollout"]) if orc else 3000
    vm = jax.jit(C.make_vm(cl.chart_map(), ell, cfg._replace(rollout_len=N)))
    fn = rec["fns"]
    out = dict(m=m, gamma=gamma, amax=cl.amax, rho=fn["V"]["level"], r_d=float(np.sqrt(max(fn["Vd"]["level"], 0.0))), r_b=cfg.eps_ball,
               spr=rec["spr"], rho_or=(orc["level"] if orc else np.nan))
    if m == 1:
        th = np.linspace(-np.pi, np.pi, 4001); zt = jp.asarray(th)[:, None]
        Vt = np.asarray(V(zt)); Ft = np.asarray(T(zt))[:, 0]
        u = np.asarray(cl.action(G.Exp(zt)))[:, 0]; idx = np.where(np.sign(u[:-1]) != np.sign(u[1:]))[0]
        out.update(th=th, V=Vt, Vm=np.asarray(vm(zt)), F=Ft, dV=np.asarray(V(T(zt))) - Vt,
                   roots=np.asarray([0.5 * (th[i] + th[i + 1]) for i in idx]))
    else:
        i1, i2 = F.SLICE[G_]; n = 321; g = np.linspace(-np.pi, np.pi, n); Z1, Z2 = np.meshgrid(g, g)
        zs = np.zeros((n * n, m)); zs[:, i1] = Z1.ravel(); zs[:, i2] = Z2.ravel(); inb = np.linalg.norm(zs, axis=-1) < np.pi - 1e-3
        Vs = np.array(V(jp.asarray(zs))); dVs = np.array(V(T(jp.asarray(zs)))) - Vs
        Vms = np.concatenate([np.asarray(vm(jp.asarray(zs[i:i + 10000]))) for i in range(0, len(zs), 10000)])
        Vs[~inb] = np.nan; dVs[~inb] = np.nan; Vms[~inb] = np.nan
        z2 = np.asarray(C.sample_ball(jax.random.key(3), 10, 2, 0.95 * gamma)); zst = np.zeros((10, m)); zst[:, i1] = z2[:, 0]; zst[:, i2] = z2[:, 1]
        Xc = G.Exp(jp.asarray(zst)); traj = [np.asarray(G.Log(Xc))]
        for _ in range(60): Xc = cl.step(Xc); traj.append(np.asarray(G.Log(Xc)))
        out.update(g=g, V=Vs.reshape(n, n), dV=dVs.reshape(n, n), Vm_grid=Vms.reshape(n, n), traj=np.stack(traj)[:, :, [i1, i2]])
    zf = np.asarray(C.sample_ball(jax.random.key(4), 4000, m, gamma))
    out.update(Vf=np.asarray(V(jp.asarray(zf))), Vmf=np.asarray(vm(jp.asarray(zf))))
    np.savez(CACHE / f"{fid}.npz", **out)
    print("cached", fid)


def load(fid):
    f = CACHE / f"{fid}.npz"
    if not f.exists(): return None
    D = dict(np.load(f)); D["gated"] = False
    return D


def oracle_overlay(ax, D):
    if int(D["m"]) == 1 or not np.isfinite(float(D["rho_or"])): return
    g = D["g"]; G1, G2 = np.meshgrid(g, g)
    M = ((np.hypot(G1, G2) <= float(D["gamma"])) & (np.nan_to_num(D["Vm_grid"], nan=np.inf) <= float(D["rho_or"]))).astype(float)
    if M.any(): ax.contour(g, g, M, levels=[0.5], colors=F.INK, linewidths=0.8, linestyles="dotted")


def blank_row(axs, text):
    for a in axs: a.axis("off")
    axs[1].text(0.5, 0.5, text, ha="center", va="center", fontsize=6.5, color=F.INK, transform=axs[1].transAxes)


def spr_of_failed(G_, algo, seed):
    tag = "" if algo == "ppo" else f"_{algo}"
    f = RES / "l4dc" / f"roa_{G_}{tag}_seed{seed}.json"
    return json.load(open(f))["fixed_points"]["rho_goal"] if f.exists() else None


def figure(rows, labels, path, W=6.3, font_scale=1.0, hspace=0.42, wspace=None, label_x=None, shared_xlabel=False):
    """font_scale multiplies the size of every text element (titles, labels, tick labels, colour-bar ticks, row labels)."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    so2 = all(r[0].startswith("SO2") for r in rows if r[0])
    h = 1.55 if so2 else 1.95
    fig, axs = plt.subplots(len(rows), 3, figsize=(W, len(rows) * h), squeeze=False,
                            gridspec_kw=dict(wspace=wspace if wspace is not None else (0.3 if so2 else 0.42), hspace=hspace))
    for k, (fid, lab) in enumerate(zip(rows, labels)):
        D = load(fid[0]) if fid else None
        if D is None:
            blank_row(axs[k], fid[1] if fid else ""); continue
        F.row(axs[k], D, fid[0].split("__")[0], k == 0, k == len(rows) - 1)
        oracle_overlay(axs[k][0], D)
        axs[k, 0].annotate(lab, xy=(label_x if label_x is not None else (-0.42 if not so2 else -0.3), 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center", fontsize=7)
    if shared_xlabel:
        for k in range(len(rows) - 1):
            for a in axs[k]: a.set_xlabel("")
    if font_scale != 1.0:
        for t in fig.findobj(Text):
            t.set_fontsize(t.get_fontsize() * font_scale)
    fig.savefig(path, dpi=300); plt.close(fig); print("drew", path)


def draw(outdir):
    F.style(); outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for G_ in F.GROUPS:
        for algo in F.ALGOS:
            rows, labels = [], []
            for s in range(3):
                fid = f"{G_}__{algo}_{G_}_ls-1.5__s{s}__base"
                if load(fid) is not None:
                    rows.append((fid, None))
                else:
                    spr = spr_of_failed(G_, algo, s)
                    rows.append((None, f"spr(DF(0)) = {spr:.3g}: fails the stability test" if spr and spr > 0.99 else "not yet computed"))
                labels.append(f"seed {s}")
            if all(r[0] is None for r in rows): continue
            figure(rows, labels, outdir / f"fig_{G_}_{algo}.pdf")
    main = [("SO3__td3_SO3_ls-1.5__s1__base", None), ("SE3__ppo_SE3_ls-1.5__s1__base", None)]
    figure(main, [r"$\mathrm{SO}(3)$, TD3 seed 1", r"$\mathrm{SE}(3)$, PPO seed 1"], outdir / "fig_main.pdf")
    pert = sorted(p.stem for p in CACHE.glob("*.npz") if not p.stem.endswith("__base"))
    lab = lambda i: (lambda G_, d, s, v: f"{F.GLAB[G_]}, {F.ALAB[d.split('_')[0]]} {s[1:]}, "
                     + (f"gain {v[1:]}" if v.startswith("g") else f"{v[1:]}" + r"$^\circ$"))(*i.split("__"))
    for k in range(0, len(pert), 6):
        chunk = pert[k:k + 6]
        figure([(i, None) for i in chunk], [lab(i) for i in chunk], outdir / f"fig_perturbed_{k // 6 + 1}.pdf")


def draw_main(path, font_scale=1.5, hspace=0.42, wspace=None, label_x=None, shared_xlabel=0):
    """Figure 1 of the paper alone, with scaled text and adjustable spacing."""
    F.style()
    main = [("SO3__td3_SO3_ls-1.5__s1__base", None), ("SE3__ppo_SE3_ls-1.5__s1__base", None)]
    figure(main, [r"$\mathrm{SO}(3)$, TD3 seed 1", r"$\mathrm{SE}(3)$, PPO seed 1"], Path(path),
           font_scale=font_scale, hspace=hspace, wspace=wspace, label_x=label_x, shared_xlabel=bool(shared_xlabel))


if __name__ == "__main__":
    if sys.argv[1] == "data": data(sys.argv[2])
    elif sys.argv[1] == "main": draw_main(sys.argv[2], *[float(x) for x in sys.argv[3:]])
    else: draw(sys.argv[2])
