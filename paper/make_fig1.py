"""Figure 1 (new): the region of attraction of one policy before and after an actuator misalignment, in 3D."""
import json, os, pickle, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

REPO = Path(__file__).resolve().parents[1] / "lie_rl"
RES = REPO / "results"
OUT = Path(__file__).resolve().parent / "out" / "fig1"
CACHE = Path(__file__).resolve().parent / "out" / "fig1" / "fig1_data.npz"
CACHE.parent.mkdir(parents=True, exist_ok=True)
FID = "SE2__ppo_SE2_ls-1.5__s0__m40"
BASE = "SE2__ppo_SE2_ls-1.5__s0__base"
NPOL, NAZ = 49, 97
NPOL_O, NAZ_O = 25, 49


def rays(npol, naz):
    pol = np.linspace(0, np.pi, npol); az = np.linspace(-np.pi, np.pi, naz)
    P, A = np.meshgrid(pol, az, indexing="ij")
    d = np.stack([np.cos(P), np.sin(P) * np.cos(A), np.sin(P) * np.sin(A)], -1)
    return P, A, d


def data():
    sys.path.insert(0, str(REPO))
    import jax, jax.numpy as jp
    from lie_rl import certificate as C, anchor as AN, roa as R, variants as Vr
    t0 = time.time()
    cl = Vr.ClosedLoop("SE2", "ppo_SE2_ls-1.5", 0, "m40"); G = cl.G; m = cl.m
    T = cl.chart_map(); Tj = jax.jit(T)
    ell = C.make_ell(G)

    def cand(path):
        cert = pickle.load(open(path, "rb")); cfg = C.CertConfig(**cert["cfg"])
        p = C.CertParams(psi=[(jp.asarray(w), jp.asarray(b)) for w, b in cert["psi"]],
                         alpha_raw=jp.asarray(cert["alpha_raw"]), eps_raw=jp.asarray(cert["eps_raw"]))
        Vc = AN.anchored(np.asarray(cert["anchor"]["P"]), cert["anchor"]["r_a"])
        return jax.jit(lambda z: Vc(p, cfg, ell, z)), cfg

    V40, cfg = cand(RES / "final" / "cert" / f"{FID}.pkl")
    V0, _ = cand(RES / "final" / "cert" / f"{BASE}.pkl")
    gamma, r_b = cfg.radius, cfg.eps_ball
    ev = json.load(open(RES / "final" / "eval" / f"{FID}.json")); ev0 = json.load(open(RES / "final" / "eval" / f"{BASE}.json"))
    orc = json.load(open(RES / "rx" / "oracle" / f"{FID}.json")); nom = json.load(open(RES / "review" / "nominal" / f"{FID}.json"))
    rho = ev["fns"]["V"]["level"]; rho_d = ev["fns"]["Vd"]["level"]; rho_P = ev["fns"]["VP"]["level"]
    rho0 = ev0["fns"]["V"]["level"]; rho_or = orc["level"]; N_or = int(orc["N_rollout"]); RHO_CACHE["rho"] = rho
    assert abs(rho0 - nom["nominal_level"]) < 1e-9
    A = R.jacobian_at_zero(T, m); P = R.stein(A)
    assert abs(np.max(np.abs(np.linalg.eigvals(A))) - ev["spr"]) < 1e-9

    def evalb(fn, z, bs=50000):
        return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])

    @jax.jit
    def vm_long(z):
        def body(c, _):
            X, acc = c
            return (cl.step(X), acc + jp.sum((G.obs(X) - G.obs(G.identity(1))) ** 2, axis=-1)), None
        (Xf, acc), _ = jax.lax.scan(body, (G.Exp(z), jp.zeros(z.shape[0])), None, length=N_or)
        th, tr = G.err_parts(Xf)
        return jp.where(jp.maximum(th, tr) < 0.05, acc, jp.inf)

    def vm(z): return evalb(vm_long, z, bs=5000)

    def ray_radius(fn, level, d, nr, nbis):
        n = len(d); r = np.linspace(gamma / nr, gamma, nr)
        pts = (d[:, None, :] * r[None, :, None]).reshape(-1, 3)
        inside = (fn(pts) <= level).reshape(n, nr)
        first_out = np.where(~inside.all(1), np.argmin(inside, 1), nr)
        later_in = np.array([inside[i, first_out[i]:].any() for i in range(n)])
        lo = np.where(first_out > 0, r[np.maximum(first_out - 1, 0)], 0.0); hi = np.where(first_out < nr, r[np.minimum(first_out, nr - 1)], gamma)
        full = first_out == nr
        for _ in range(nbis):
            mid = 0.5 * (lo + hi); ok = fn(d * mid[:, None]) <= level
            lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
        rad = np.where(full, gamma, 0.5 * (lo + hi))
        return rad, later_in, inside

    def vol_frac(rad, Pp):
        w = np.sin(Pp); return float(np.sum(w * (rad / gamma) ** 3) / np.sum(w))

    if os.environ.get("FIG1_ONLY_SPHERE"):
        old = dict(np.load(CACHE)); old.update(sphere_fields(V40, vm, ray_radius, gamma)); np.savez(CACHE, **old); return
    Pp, Aa, d = rays(NPOL, NAZ); dd = d.reshape(-1, 3)
    out = dict(gamma=gamma, r_b=r_b, rho=rho, rho_d=rho_d, rho_P=rho_P, rho0=rho0, rho_or=rho_or, P=P, A=A, pol=Pp, az=Aa)
    checks = {}
    for name, fn, lev in [("V", lambda z: evalb(V40, z), rho), ("V0", lambda z: evalb(V0, z), rho0)]:
        rad, bad, _ = ray_radius(fn, lev, dd, 400, 30)
        out[f"r_{name}"] = rad.reshape(Pp.shape)
        checks[name] = dict(nonstar_rays=int(bad.sum()), rays=len(dd), radii_per_ray=400, vol_frac_rays=vol_frac(out[f"r_{name}"], Pp))
        print(name, checks[name], f"{time.time() - t0:.0f}s", flush=True)
    r_d = np.full(len(dd), np.sqrt(rho_d)); r_P = np.sqrt(rho_P / np.einsum("ni,ij,nj->n", dd, P, dd))
    out["r_U"] = np.minimum(np.maximum(r_d, r_P), gamma).reshape(Pp.shape)
    checks["union"] = dict(vol_frac_rays=vol_frac(out["r_U"], Pp))
    Po, Ao, do_ = rays(NPOL_O, NAZ_O); ddo = do_.reshape(-1, 3)
    rad, bad, _ = ray_radius(vm, rho_or, ddo, 36, 10)
    out["r_O"] = rad.reshape(Po.shape); out["pol_O"] = Po; out["az_O"] = Ao
    checks["oracle"] = dict(nonstar_rays=int(bad.sum()), rays=len(ddo), radii_per_ray=36, vol_frac_rays=vol_frac(out["r_O"], Po))
    print("oracle", checks["oracle"], f"{time.time() - t0:.0f}s", flush=True)

    zu = np.asarray(C.sample_ball(jax.random.key(11), 200000, m, gamma))
    checks["V"]["vol_frac_sampled_eval"] = ev["fns"]["V"]["region_ball_fraction"]
    checks["V"]["vol_frac_sampled_here"] = float(np.mean(evalb(V40, zu) <= rho))
    checks["V0"]["vol_frac_sampled_eval"] = ev0["fns"]["V"]["region_ball_fraction"]
    checks["V0"]["vol_frac_sampled_here"] = float(np.mean(evalb(V0, zu) <= rho0))
    checks["union"]["vol_frac_sampled_eval"] = ev["unions"]["Vd|VP"]["region_ball_fraction"]
    checks["union"]["vol_frac_sampled_here"] = float(np.mean((np.sum(zu ** 2, 1) <= rho_d) | (np.einsum("ni,ij,nj->n", zu, P, zu) <= rho_P)))
    checks["oracle"]["vol_frac_sampled_here"] = float(np.mean(vm(zu[:12000]) <= rho_or))
    print(checks, flush=True)

    z = np.asarray(C.sample_ball(jax.random.key(20261018), 1000000, m, gamma))
    v = evalb(V0, z); Fz = evalb(Tj, z); vF = evalb(V0, Fz); l = evalb(ell, z)
    inreg = v <= rho0; outnb = l >= r_b
    dec_bad = inreg & outnb & (vF - v >= 0); cont_bad = inreg & (np.linalg.norm(Fz, axis=-1) > gamma)
    checks["nominal_check"] = dict(decrease=int(dec_bad.sum()), containment=int(cont_bad.sum()),
                                   json_decrease=nom["decrease_violations"], json_containment=nom["containment_violations"],
                                   frac=float((dec_bad | cont_bad).sum() / (inreg & outnb).sum()), json_frac=nom["violation_fraction_of_region"])
    fail = dec_bad | cont_bad; zf = z[fail]
    checks["nominal_check"]["fail_outside_new_region"] = float(np.mean(evalb(V40, zf) > rho))
    print(checks["nominal_check"], flush=True)
    out["z_fail"] = zf; out["fail_kind"] = cont_bad[fail].astype(int)

    out["checks"] = json.dumps(checks)
    out.update(sphere_fields(V40, vm, ray_radius, gamma))
    np.savez(CACHE, **out); print("cached", CACHE, f"{time.time() - t0:.0f}s")


NPOL_F, NAZ_F = 97, 193
NPOL_S, NAZ_S = 193, 385
NPOL_SO, NAZ_SO = 49, 97


def sphere_fields(V40, vm, ray_radius, gamma):
    """V and V_m on the sphere |z| = gamma (their level curves there are the rims of the regions on the ball's"""
    import jax.numpy as jp
    def evalb(fn, z, bs=50000):
        return np.concatenate([np.asarray(fn(jp.asarray(z[i:i + bs]))) for i in range(0, len(z), bs)])
    out = {}
    P, A, d = rays(NPOL_S, NAZ_S); out["sV"] = evalb(V40, gamma * d.reshape(-1, 3)).reshape(P.shape); out["pol_S"], out["az_S"] = P, A
    P, A, d = rays(NPOL_SO, NAZ_SO); out["sO"] = vm(gamma * d.reshape(-1, 3)).reshape(P.shape); out["pol_SO"], out["az_SO"] = P, A
    P, A, d = rays(NPOL_F, NAZ_F); dd = d.reshape(-1, 3); rad = []; bad = 0
    for i in range(0, len(dd), 2000):
        r_, b_, _ = ray_radius(lambda z: evalb(V40, z), RHO_CACHE["rho"], dd[i:i + 2000], 400, 30); rad.append(r_); bad += int(b_.sum())
    out["r_Vf"] = np.concatenate(rad).reshape(P.shape); out["pol_F"], out["az_F"] = P, A; out["nonstar_fine"] = bad
    print("sphere fields; non-star rays on the fine grid:", bad, flush=True)
    return out


RHO_CACHE = {}


def loops26():
    ids = []
    for f in sorted((RES / "final" / "eval").glob("*.json")):
        i = f.stem
        if not i.endswith("__base") and i.split("__")[-1] not in ("m50", "m60"): ids.append((i, f))
    for f in sorted((RES / "review" / "pert36" / "eval").glob("*__final.json")):
        i = f.stem[:-len("__final")]
        if i.split("__")[-1] not in ("m50", "m60"): ids.append((i, f))
    rows = []
    for i, f in ids:
        ev = json.load(open(f)); orc = json.load(open(RES / "rx" / "oracle" / f"{i}.json"))
        v = ev["fns"]["V"]["region_mu0"]; u = ev["unions"]["Vd|VP"]["region_mu0"]; o = orc["region_mu0"]
        rows.append((i, 100 * v, 100 * u, 100 * o))
    assert len(rows) == 26, len(rows)
    return rows


def loops55():
    """The 55 evaluated closed loops with an actuator error: the 26 of the first screen and the 29 of the later screens."""
    rows = [r for r in loops26()]
    for blk in ("pert_r2", "pert_r3"):
        ids = [l.strip() for l in open(RES / "review" / ("perturbed_r2_ids.txt" if blk == "pert_r2" else "perturbed_r3_ids_kept.txt")) if l.strip()]
        for i in ids:
            ev = json.load(open(RES / "review" / blk / "eval" / f"{i}__final.json")); orc = json.load(open(RES / "rx" / "oracle" / f"{i}.json"))
            rows.append((i, 100 * ev["fns"]["V"]["region_mu0"], 100 * ev["unions"]["Vd|VP"]["region_mu0"], 100 * orc["region_mu0"]))
    assert len(rows) == 55 and len({r[0] for r in rows}) == 55, len(rows)
    return rows


def check_reference55(rows):
    v = np.array([r[1] for r in rows]); u = np.array([r[2] for r in rows]); o = np.array([r[3] for r in rows])
    rv, ru = 100 * v / o, 100 * u / o
    out = dict(n=len(rows), margin_min=float((v - u).min()), margin_max=float((v - u).max()), rv_median=float(np.median(rv)),
               ru_median=float(np.median(ru)), rv_min=float(rv.min()), ru_min=float(ru.min()), ru_max=float(ru.max()), n99=int((rv >= 99).sum()))
    assert round(out["margin_min"], 1) == 5.4 and round(out["margin_max"], 1) == 51.8, out
    assert round(out["rv_median"], 1) == 99.9 and round(out["ru_median"], 1) == 62.8 and out["n99"] == 37, out
    assert round(out["rv_min"], 1) == 66.3 and round(out["ru_min"], 1) == 12.8 and round(out["ru_max"], 1) == 93.4, out
    return out


def check_reference(rows):
    v = np.array([r[1] for r in rows]); u = np.array([r[2] for r in rows]); o = np.array([r[3] for r in rows])
    rv, ru = 100 * v / o, 100 * u / o
    out = dict(n=len(rows), margin_min=float((v - u).min()), margin_max=float((v - u).max()),
               rv_median=float(np.median(rv)), ru_median=float(np.median(ru)), rv_min=float(rv.min()), rv_max=float(rv.max()),
               ru_min=float(ru.min()), ru_max=float(ru.max()))
    assert round(out["margin_min"], 1) == 5.4 and round(out["margin_max"], 1) == 51.8, out
    assert round(out["rv_median"], 1) == 99.8 and round(out["ru_median"], 1) == 58.2, out
    assert round(out["rv_min"], 1) == 69.0 and round(out["ru_min"], 1) == 12.8 and round(out["ru_max"], 1) == 93.4, out
    return out


C_CAND, C_BASE, C_FAIL = "#0072B2", "#E69F00", "#C8102E"
C_OLD = "#8fbbd9"
C_EXCL = "#9a9a9a"
INK, INK2, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def cam_vec(elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


class Radius:
    """Boundary radius r(direction) of a star-shaped region from its ray grid (bilinear in polar/azimuth)."""
    def __init__(self, R, pol, az):
        from scipy.interpolate import RegularGridInterpolator
        self.R, self.pol, self.az = R, pol, az
        self.f = RegularGridInterpolator((pol[:, 0], az[0]), R)

    def __call__(self, d):
        d = np.atleast_2d(d)
        p = np.arccos(np.clip(d[:, 0], -1, 1)); a = np.arctan2(d[:, 2], d[:, 1])
        return self.f(np.stack([p, a], -1))


def in_cut(c, cut):
    return np.all(np.stack([sgn * c[..., k] > 1e-9 for k, sgn in enumerate(cut)], -1), -1)


def surface_quads(rad, cut, keep=None):
    """Quads of the surface r = rad(direction) on rad's grid; faces inside the cut octant removed."""
    pol, az, R = rad.pol, rad.az, rad.R
    d = np.stack([np.cos(pol), np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az)], -1)
    X = R[..., None] * d
    q = np.stack([X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]], 2)
    cdir = (d[:-1, :-1] + d[1:, :-1] + d[1:, 1:] + d[:-1, 1:]); cdir /= np.linalg.norm(cdir, axis=-1, keepdims=True)
    m = np.ones(q.shape[:2], bool) if cut is None else ~in_cut(cdir, cut)
    if keep is not None: m &= keep
    q = q[m]; c = q.mean(1)
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nn > 1e-12, n / np.maximum(nn, 1e-12), cdir[m]); n *= np.sign(np.sum(n * c, -1, keepdims=True) + 1e-12)
    return q, n


def cap_quads(r_in, r_out, cut, nt=40):
    """Annular faces between radii r_in and r_out on the three quarter-planes bounding the cut octant."""
    quads, normals = [], []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        t = np.linspace(0, np.pi / 2, nt)
        d = np.zeros((nt, 3)); d[:, i] = cut[i] * np.cos(t); d[:, j] = cut[j] * np.sin(t)
        a = r_in(d) if callable(r_in) else np.full(nt, float(r_in)); b = r_out(d)
        P0, P1 = a[:, None] * d, b[:, None] * d
        for s in range(nt - 1):
            if b[s] - a[s] < 1e-6 and b[s + 1] - a[s + 1] < 1e-6: continue
            quads.append(np.array([P0[s], P1[s], P1[s + 1], P0[s + 1]]))
            nv = np.zeros(3); nv[k] = cut[k]; normals.append(nv)
    return np.array(quads).reshape(-1, 4, 3), np.array(normals).reshape(-1, 3)


def shade(rgb, n, light, amb=0.45, dif=0.6, spec=0.0, cam=None):
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(rgb))
    lam = np.clip(n @ light, 0, 1)
    out = base[None] * (amb + dif * lam[:, None])
    if spec and cam is not None:
        h = light + cam; h /= np.linalg.norm(h); out += spec * np.clip(n @ h, 0, 1)[:, None] ** 30
    return np.clip(out, 0, 1)


def silhouette(ax, gamma, elev, azim, **kw):
    c = cam_vec(elev, azim); u = np.cross(c, [0, 0, 1.0]); u /= np.linalg.norm(u); w = np.cross(c, u)
    t = np.linspace(0, 2 * np.pi, 361); P = gamma * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))
    ax.plot(P[:, 0], P[:, 1], P[:, 2], **kw)


def triad(ax, origin, L, labels, fs=8):
    for k in range(3):
        e = np.zeros(3); e[k] = L
        ax.plot(*[[origin[i], origin[i] + e[i]] for i in range(3)], color=INK2, lw=0.6, zorder=50)
        ax.text(*(origin + 1.4 * e), labels[k], fontsize=fs, ha="center", va="center", color=INK, zorder=51)


def ax3d(fig, rect, gamma, elev, azim):
    ax = fig.add_axes(rect, projection="3d", proj_type="ortho")
    ax.view_init(elev=elev, azim=azim)
    lim = 1.02 * gamma
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim); ax.set_box_aspect((1, 1, 1), zoom=1.0)
    ax.set_axis_off(); ax.computed_zorder = False; ax.patch.set_alpha(0.0)
    return ax


def to2d(ax, p):
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj()); return x, y


def lighten(c, f):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(c)); return tuple(1 - (1 - c) * f)


def screen_axes(elev, azim):
    c = cam_vec(elev, azim); u = np.cross([0, 0, 1.0], c); u /= np.linalg.norm(u); w = np.cross(c, u)
    return c, u, w


def cap_outline(rad, cut, nt=120):
    segs = []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]; t = np.linspace(0, np.pi / 2, nt)
        d = np.zeros((nt, 3)); d[:, i] = cut[i] * np.cos(t); d[:, j] = cut[j] * np.sin(t)
        segs.append(rad(d)[:, None] * d)
    return segs


def refine(rad, f=2):
    """The same radius map on an f-times finer grid (bilinear), for smoother shading of the drawn surface."""
    pol = np.linspace(0, np.pi, (rad.pol.shape[0] - 1) * f + 1); az = np.linspace(-np.pi, np.pi, (rad.az.shape[1] - 1) * f + 1)
    P, A = np.meshgrid(pol, az, indexing="ij")
    d = np.stack([np.cos(P), np.sin(P) * np.cos(A), np.sin(P) * np.sin(A)], -1).reshape(-1, 3)
    R = rad.f(np.stack([P.ravel(), A.ravel()], -1)).reshape(P.shape)
    return Radius(R, P, A)


def sphere_grid_quads(pol, az, g):
    d = np.stack([np.cos(pol), np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az)], -1) * g
    return np.stack([d[:-1, :-1], d[1:, :-1], d[1:, 1:], d[:-1, 1:]], 2)


def level_curves(F, pol, az, level, g, cam, cut):
    """Level curve {F = level} of a function F on the sphere |z| = g, front-facing and outside the cut octant."""
    import matplotlib.pyplot as plt
    f = plt.figure(); a = f.add_subplot()
    cs = a.contour(az, pol, np.where(np.isfinite(F), F, 1e6), levels=[level]); plt.close(f)
    out = []
    for seg in cs.allsegs[0]:
        az_, pol_ = seg[:, 0], seg[:, 1]
        P = g * np.stack([np.cos(pol_), np.sin(pol_) * np.cos(az_), np.sin(pol_) * np.sin(az_)], -1)
        hide = (P @ cam <= 0) | in_cut(P, cut)
        out.append(np.where(hide[:, None], np.nan, P))
    return out


def solids(ax, D, elev, azim, cut, kind):
    """kind 'old': the region computed without the error (translucent); 'new': candidate, baseline, oracle."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgba
    import matplotlib.patheffects as pe
    g = float(D["gamma"]); cam, u, w = screen_axes(elev, azim); light = cam_vec(elev + 25, azim + 35)
    Q, C = [], []
    def add(q, rgb, a): Q.append(q); C.append(np.c_[rgb, np.full(len(q), a)])
    if kind == "old":
        r0 = refine(Radius(D["r_V0"], D["pol"], D["az"]))
        q, n = surface_quads(r0, None); n = q.mean(1) / np.linalg.norm(q.mean(1), axis=1, keepdims=True)
        front = n @ cam > 0
        add(q[front], shade(C_OLD, n[front], light, amb=0.55, dif=0.55, spec=0.2, cam=cam), 1.0)
    else:
        rV = Radius(D["r_Vf"], D["pol_F"], D["az_F"]); rU = Radius(D["r_U"], D["pol"], D["az"])
        q, n = surface_quads(rV, cut); add(q, shade(C_CAND, n, light, amb=0.5, dif=0.6, spec=0.2, cam=cam), 1.0)
        q, n = cap_quads(rU, rV, cut, nt=60); add(q, np.tile(lighten(C_CAND, 0.78), (len(q), 1)), 1.0)
        q, n = cap_quads(0.0, rU, cut, nt=60); add(q, np.tile(lighten(C_BASE, 0.8), (len(q), 1)), 1.0)
        d = np.stack([np.cos(rU.pol), np.sin(rU.pol) * np.cos(rU.az), np.sin(rU.pol) * np.sin(rU.az)], -1)
        cd = d[:-1, :-1] + d[1:, :-1] + d[1:, 1:] + d[:-1, 1:]
        q, n = surface_quads(rU, None, keep=in_cut(cd, cut)); add(q, shade(C_BASE, n, light, amb=0.45, dif=0.65), 1.0)
    Q = np.concatenate(Q); Cc = np.concatenate(C)
    ax.add_collection3d(Poly3DCollection(Q, facecolors=Cc, edgecolors=Cc, linewidths=0.15, zorder=5, rasterized=True))
    if kind == "new":
        for P in cap_outline(rV, cut): ax.plot(*P.T, color=INK, lw=0.5, zorder=20)
        for k in range(3):
            e_ = np.zeros(3); e_[k] = cut[k]; ax.plot(*np.stack([0.0 * e_, rV(e_[None])[0] * e_]).T, color=INK, lw=0.5, zorder=20)
        for P in cap_outline(rU, cut): ax.plot(*P.T, color=INK, lw=0.35, zorder=20)
        rO = Radius(D["r_O"], D["pol_O"], D["az_O"])
        for P in cap_outline(rO, cut):
            P = np.where((np.linalg.norm(P, axis=1) < 0.995 * g)[:, None], P, np.nan)
            ax.plot(*P.T, color="black", lw=1.1, ls=(0, (1.0, 1.4)), zorder=27, path_effects=HALO)
        q = sphere_grid_quads(D["pol_S"], D["az_S"], g); sV = D["sV"]
        outside = (sV[:-1, :-1] + sV[1:, :-1] + sV[1:, 1:] + sV[:-1, 1:]) / 4 > float(D["rho"])
        c = q.mean(2); keep = outside & (c @ cam > 0) & ~in_cut(c, cut)
        ax.add_collection3d(Poly3DCollection(q[keep], facecolors=to_rgba(C_EXCL, GHOST_ALPHA), edgecolors="none", zorder=22, rasterized=True))
        for P in level_curves(D["sV"], D["pol_S"], D["az_S"], float(D["rho"]), g, cam, cut): ax.plot(*P.T, color=INK, lw=0.5, zorder=25)
        for P in level_curves(D["sO"], D["pol_SO"], D["az_SO"], float(D["rho_or"]), g, cam, cut):
            ax.plot(*P.T, color="black", lw=1.1, ls=(0, (1.0, 1.4)), zorder=27, path_effects=HALO)
    silhouette(ax, g, elev, azim, color=MUTED, lw=0.6, zorder=30)


GHOST_ALPHA = 0.8


def _halo():
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=2.1, foreground="white"), pe.Normal()]


HALO = _halo()


def corner_triad(ax, g, elev, azim, L=0.75, pos=(-1.0, -1.0)):
    cam, u, w = screen_axes(elev, azim)
    o = g * (pos[0] * u + pos[1] * w)
    triad(ax, o, L, ["$z_1$", "$z_2$", "$z_3$"], fs=8)


VIEW = dict(elev=45, azim=-150, cut=(-1, -1, 1))


def draw(path_stem, dpi=300, W=6.3, H=2.0):
    import matplotlib.pyplot as plt
    style()
    D = dict(np.load(CACHE)); g = float(D["gamma"])
    rows = loops26(); led = check_reference(rows)
    e, a, cut = VIEW["elev"], VIEW["azim"], VIEW["cut"]
    cam, u, w = screen_axes(e, a)
    fig = plt.figure(figsize=(W, H))
    ZOOM, RECT_Y, RECT_H = 1.32, -0.04, 1.0
    axa = ax3d(fig, [0.0, RECT_Y, 0.33, RECT_H], g, e, a); axa.set_box_aspect((1, 1, 1), zoom=ZOOM)
    solids(axa, D, e, a, cut, "old")
    zf = D["z_fail"]; sel = np.random.default_rng(0).choice(len(zf), 3000, replace=False); zs = zf[sel]
    dep = np.clip((zs @ cam / g - 0.2) / 0.7, 0, 1)
    order = np.argsort(dep)
    from matplotlib.colors import to_rgba
    rgba = np.array([to_rgba(C_FAIL)] * len(zs)); rgba[:, 3] = 0.35 + 0.6 * dep
    axa.scatter(*zs[order].T, s=(0.5 + 1.1 * dep[order]), c=rgba[order], lw=0, depthshade=False, rasterized=True, zorder=40)
    axb = ax3d(fig, [0.33, RECT_Y, 0.33, RECT_H], g, e, a); axb.set_box_aspect((1, 1, 1), zoom=ZOOM)
    solids(axb, D, e, a, cut, "new")
    corner_triad(axb, g, e, a, pos=(-1.22, -0.95))
    fig.canvas.draw()
    axc = fig.add_axes([0.765, 0.2, 0.225, 0.66])
    v = np.array([r[1] / r[3] for r in rows]) * 100; ub = np.array([r[2] / r[3] for r in rows]) * 100
    o = np.argsort(ub); x = np.arange(len(rows))
    for k, j in enumerate(o):
        axc.plot([k, k], [ub[j], v[j]], color="#cfcfcf", lw=0.8, zorder=1, solid_capstyle="butt")
    axc.scatter(x, ub[o], s=8, color=C_BASE, lw=0, zorder=2, label="baseline")
    axc.scatter(x, v[o], s=8, color=C_CAND, lw=0, zorder=3, label="candidate")
    ex = [k for k, j in enumerate(o) if rows[j][0] == FID][0]
    axc.plot([ex, ex], [ub[o][ex], v[o][ex]], color=INK2, lw=0.9, zorder=1.5, solid_capstyle="butt")
    axc.annotate("loop of (b)", xy=(ex + 0.15, ub[o][ex] - 3), xytext=(ex + 2.2, 33), ha="left", va="center", fontsize=7,
                 color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0.5, shrinkB=0.5))
    axc.set_ylim(0, 104); axc.set_yticks([0, 50, 100]); axc.set_xticks([]); axc.set_xlim(-1, len(rows))
    axc.set_ylabel(r"% of oracle region", labelpad=2)
    axc.set_xlabel("closed loop", labelpad=3)
    for s_ in ("top", "right"): axc.spines[s_].set_visible(False)
    axc.legend(loc="lower right", frameon=False, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25, markerscale=1.2)
    def fig_xy(ax, p3):
        x, y = to2d(ax, p3); return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    rV = Radius(D["r_Vf"], D["pol_F"], D["az_F"]); rU = Radius(D["r_U"], D["pol"], D["az"])
    dB = np.array(cut, float) / np.sqrt(3); pB = 0.97 * rU(dB)[0] * dB
    dC = 0.45 * u - 0.75 * w + 0.5 * cam; dC /= np.linalg.norm(dC); pC = rV(dC)[0] * dC
    rimO = [P[np.isfinite(P).all(1)] for P in level_curves(D["sO"], D["pol_SO"], D["az_SO"], float(D["rho_or"]), g, cam, cut)]
    rimO = np.concatenate([P for P in rimO if len(P)]); pO = rimO[np.argmax(rimO @ u)]
    capdir = cap_direction(D); un = zf / np.linalg.norm(zf, axis=1, keepdims=True)
    pF = np.median(zf[un @ capdir > 0.8], axis=0)
    def lab(ax, p3, off, text, ha):
        x, y = to2d(ax, p3); fx, fy = fig_xy(ax, p3); txy = (fx + off[0], fy + off[1])
        ax.annotate(text, xy=(x, y), xycoords="data", xytext=txy, textcoords="figure fraction", ha=ha, va="center",
                    fontsize=8, color=INK, zorder=60, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1.0, shrinkB=0.0))
    P_, A_ = D["pol_S"], D["az_S"]; dS = np.stack([np.cos(P_), np.sin(P_) * np.cos(A_), np.sin(P_) * np.sin(A_)], -1)
    capm = (D["sV"] > float(D["rho"])) & (dS @ cam > 0.2) & ~in_cut(dS, cut); pts = g * dS[capm]
    pX = pts[np.argmin((pts @ u - np.median(pts @ u)) ** 2 + (pts @ w - np.median(pts @ w)) ** 2)]
    LBL = dict(fail=(0.06, 0.07), oracle=(0.04, 0.08), baseline=(-0.15, -0.16), candidate=(0.05, -0.05), excl=(-0.10, 0.07))
    lab(axa, pF, LBL["fail"], "fails", "left")
    lab(axb, pO, LBL["oracle"], "oracle", "left")
    lab(axb, pB, LBL["baseline"], "baseline", "right")
    lab(axb, pX, LBL["excl"], "excluded", "right")
    lab(axb, pC, LBL["candidate"], "candidate", "left")
    for x_, t in ((0.165, "(a) Not recomputed"), (0.495, "(b) Recomputed"), (0.875, "(c) All actuator errors")):
        fig.text(x_, 0.985, t, ha="center", va="top", fontsize=8)
    fig.savefig(f"{path_stem}.png", dpi=dpi); fig.savefig(f"{path_stem}.pdf", dpi=dpi)
    plt.close(fig)
    return led


def cap_direction(D):
    """Area-weighted mean direction of the part of the sphere |z| = gamma outside the recomputed region."""
    P, A = D["pol_S"], D["az_S"]; d = np.stack([np.cos(P), np.sin(P) * np.cos(A), np.sin(P) * np.sin(A)], -1)
    m = D["sV"] > float(D["rho"]); c = (d[m] * np.sin(P)[m][:, None]).sum(0); return c / np.linalg.norm(c)


def view_search():
    """For each camera and cut octant: distance on the page between the median failing point of (a) and the centre of"""
    D = dict(np.load(CACHE)); g = float(D["gamma"]); zf = D["z_fail"]
    P, A = D["pol_S"], D["az_S"]; d = np.stack([np.cos(P), np.sin(P) * np.cos(A), np.sin(P) * np.sin(A)], -1)
    m = D["sV"] > float(D["rho"]); wgt = np.sin(P); res = []
    for e in range(20, 75, 5):
        for a in range(-180, 180, 10):
            cam, u, w = screen_axes(e, a); vis = m & (d @ cam > 0)
            if not vis.any(): continue
            ww = (wgt * (d @ cam))[vis]
            cs = np.array([(d[vis] @ u * ww).sum(), (d[vis] @ w * ww).sum()]) / ww.sum()
            fs = np.array([np.median(zf @ u), np.median(zf @ w)]) / g
            for cut in [(sx, sy, 1) for sx in (-1, 1) for sy in (-1, 1)]:
                res.append((float(np.linalg.norm(cs - fs)), e, a, cut, round(float(np.array(cut) @ cam / np.sqrt(3)), 2),
                            round(float((wgt * in_cut(d, cut))[m].sum() / wgt[m].sum()), 2)))
    res.sort(key=lambda r: r[0])
    for r in [r for r in res if r[4] > 0.45 and r[5] < 0.15][:12]: print(r)


if __name__ == "__main__":
    if sys.argv[1] == "data": data()
    elif sys.argv[1] == "checks": print(json.dumps(json.loads(str(np.load(CACHE)["checks"])), indent=1))
    elif sys.argv[1] == "views": view_search()
    elif sys.argv[1] == "draw":
        if not CACHE.exists(): data()
        print(draw(str(OUT / (sys.argv[2] if len(sys.argv) > 2 else "fig1_new")), dpi=int(sys.argv[3]) if len(sys.argv) > 3 else 300))
