"""Figure 1, version 4 (per-baseline: (b) draws the region of V_P, (c) plots V_d and V_P separately). Version 3: the same content and data as make_fig1.py, rendered with VTK/PyVista (layered translucent
surfaces, three-point lighting, specular highlights) and composed in matplotlib (vector text, Times/STIX).
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig1 as M

SCR = HERE / "out" / "fig1"
SCR.mkdir(parents=True, exist_ok=True)
CUT = (-1, -1, 1)
ELEV, AZIM = 45, -150
PX = 1600
SCALE = 1.04

C_CAND = "#2d7fd3"; C_CAND_CUT = "#cfe3f6"; C_BASE = "#f2a516"; C_FAIL = "#c81d3a"
C_VD = "#6f6f6f"; C_BALL = "#4a90da"; C_BALL_IN = "#6aa8e8"; C_EXCL = "#d3d1ce"; INK, INK2, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"


def checks():
    D = dict(np.load(M.CACHE)); ch = json.loads(str(D["checks"]))
    RES = M.RES; FID, BASE = M.FID, M.BASE
    ev = json.load(open(RES / "final" / "eval" / f"{FID}.json")); ev0 = json.load(open(RES / "final" / "eval" / f"{BASE}.json"))
    orc = json.load(open(RES / "rx" / "oracle" / f"{FID}.json")); nom = json.load(open(RES / "review" / "nominal" / f"{FID}.json"))
    assert float(D["rho"]) == ev["fns"]["V"]["level"] and float(D["rho_d"]) == ev["fns"]["Vd"]["level"]
    assert float(D["rho_P"]) == ev["fns"]["VP"]["level"] and float(D["rho0"]) == ev0["fns"]["V"]["level"]
    assert float(D["rho_or"]) == orc["level"] and abs(float(D["rho0"]) - nom["nominal_level"]) < 1e-9
    assert ch["V"]["nonstar_rays"] == 0 and ch["V0"]["nonstar_rays"] == 0 and ch["oracle"]["nonstar_rays"] == 0
    assert int(D["nonstar_fine"]) == 0
    def vf(R, P): w = np.sin(P); return float(np.sum(w * (R / float(D["gamma"])) ** 3) / np.sum(w))
    for k, R, P, ref in [("V", D["r_V"], D["pol"], ev["fns"]["V"]["region_ball_fraction"]),
                         ("V0", D["r_V0"], D["pol"], ev0["fns"]["V"]["region_ball_fraction"]),
                         ("union", D["r_U"], D["pol"], ev["unions"]["Vd|VP"]["region_ball_fraction"]),
                         ("Vfine", D["r_Vf"], D["pol_F"], ev["fns"]["V"]["region_ball_fraction"])]:
        assert abs(vf(R, P) - ref) < 3e-3, (k, vf(R, P), ref)
    rPv = r_base(dirs(D["pol"], D["az"]), D)
    assert abs(vf(rPv, D["pol"]) - ev["fns"]["VP"]["region_ball_fraction"]) < 3e-3, (vf(rPv, D["pol"]), ev["fns"]["VP"]["region_ball_fraction"])
    assert ev["fns"]["VP"]["region_mu0"] > ev["fns"]["Vd"]["region_mu0"]
    assert abs(vf(D["r_O"], D["pol_O"]) - ch["oracle"]["vol_frac_sampled_here"]) < 3e-3
    assert np.allclose(r_union(dirs(D["pol"], D["az"]), D), D["r_U"], atol=1e-12)
    nc = ch["nominal_check"]
    assert nc["decrease"] == nom["decrease_violations"] and nc["containment"] == nom["containment_violations"]
    assert max(nc["decrease"], nc["containment"]) <= len(D["z_fail"]) <= nc["decrease"] + nc["containment"]
    led = M.check_reference(M.loops26())
    out = dict(V=ch["V"], V0=ch["V0"], union=ch["union"], oracle=ch["oracle"], nonstar_fine=int(D["nonstar_fine"]),
               nominal=nc, n_fail=len(D["z_fail"]), reference=led, Vfine_vol=vf(D["r_Vf"], D["pol_F"]))
    print(json.dumps(out, indent=1)); print("all checks passed")
    return out


def dirs(P, A):
    return np.stack([np.cos(P), np.sin(P) * np.cos(A), np.sin(P) * np.sin(A)], -1)


def r_union(d, D):
    P = D["P"]; g = float(D["gamma"])
    rd = np.sqrt(float(D["rho_d"])); rP = np.sqrt(float(D["rho_P"]) / np.einsum("...i,ij,...j->...", d, P, d))
    return np.minimum(np.maximum(rd, rP), g)


def r_base(d, D):
    """Boundary radius of the region of V_P, {z^T P z <= rho_P}, capped at gamma (the larger baseline region on this loop)."""
    P = D["P"]; g = float(D["gamma"])
    return np.minimum(np.sqrt(float(D["rho_P"]) / np.einsum("...i,ij,...j->...", d, P, d)), g)


def loops55_pb():
    """(id, V, Vd, VP, oracle) in percent for the 55 closed loops with an actuator error (same files as make_fig1.loops55)."""
    out = []
    for r in M.loops55():
        i = r[0]
        for p in (M.RES / "review" / "pert36" / "eval" / f"{i}__final.json", M.RES / "final" / "eval" / f"{i}.json",
                  M.RES / "review" / "pert_r2" / "eval" / f"{i}__final.json", M.RES / "review" / "pert_r3" / "eval" / f"{i}__final.json"):
            if p.exists(): break
        fn = json.load(open(p))["fns"]
        assert abs(100 * fn["V"]["region_mu0"] - r[1]) < 1e-9, i
        out.append((i, r[1], 100 * fn["Vd"]["region_mu0"], 100 * fn["VP"]["region_mu0"], r[3]))
    v, d_, p_, o = (np.array([x[k] for x in out]) for k in (1, 2, 3, 4))
    assert round(float((v - d_).min()), 1) == 6.2 and round(float((v - p_).min()), 1) == 5.4
    assert round(float(np.median(100 * d_ / o)), 1) == 59.1 and round(float(np.median(100 * p_ / o)), 1) == 58.5
    return out


def cam_axes(e=ELEV, a=AZIM):
    c = M.cam_vec(e, a); u = np.cross([0, 0, 1.0], c); u /= np.linalg.norm(u); w = np.cross(c, u)
    return c, u, w


def in_cut(c):
    return M.in_cut(c, CUT)


def grid_mesh(X, drop=None):
    """Triangle mesh of an (n, m, 3) grid of points; cells with drop True left out; duplicate points merged."""
    import pyvista as pv
    n, m, _ = X.shape; idx = np.arange(n * m).reshape(n, m)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    keep = np.ones((n - 1, m - 1), bool) if drop is None else ~drop
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    mesh = pv.PolyData(X.reshape(-1, 3).copy(), np.c_[np.full(len(tri), 3), tri].ravel())
    mesh = mesh.clean(tolerance=1e-7)
    tri = mesh.faces.reshape(-1, 4)[:, 1:]
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    mesh = pv.PolyData(mesh.points, np.c_[np.full(ok.sum(), 3), tri[ok]].ravel())
    return mesh


def cell_dirs(P, A):
    d = dirs(P, A); c = d[:-1, :-1] + d[1:, :-1] + d[1:, 1:] + d[:-1, 1:]
    return c / np.linalg.norm(c, axis=-1, keepdims=True)


def fine_grid(f):
    """(f*96+1) x (f*192+1) polar/azimuth grid: contains the grids of the cache, so the octant's planes are grid lines."""
    pol = np.linspace(0, np.pi, 96 * f + 1); az = np.linspace(-np.pi, np.pi, 192 * f + 1)
    return np.meshgrid(pol, az, indexing="ij")


def outward(mesh):
    """Smooth point normals, flipped to point away from 0 on average."""
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True, split_vertices=False, consistent_normals=True,
                                auto_orient_normals=False)
    if np.sum(mesh.point_data["Normals"] * mesh.points) < 0: mesh.point_data["Normals"] *= -1
    return mesh


def quarter_plane_faces(r_in, r_out, nt=181, ns=40):
    """The three quarter-plane faces bounding the cut octant, between radii r_in(d) and r_out(d)."""
    out = []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        t = np.linspace(0, np.pi / 2, nt); d = np.zeros((nt, 3)); d[:, i] = CUT[i] * np.cos(t); d[:, j] = CUT[j] * np.sin(t)
        a = r_in(d); b = r_out(d); s = np.linspace(0, 1, ns)
        X = (a[:, None] + s[None] * (b - a)[:, None])[..., None] * d[:, None, :]
        m = grid_mesh(X); nv = np.zeros(3); nv[k] = CUT[k]
        m.point_data["Normals"] = np.tile(nv, (m.n_points, 1)); out.append(m)
    return out


def to_px(p, g, px):
    c, u, w = cam_axes(); S = SCALE * g; p = np.atleast_2d(p)
    return np.stack([px / 2 + (p @ u) / S * px / 2, px / 2 - (p @ w) / S * px / 2], -1)


LIGHTS = [((-1.0, 1.3, 1.2), "#fff4e6", 0.80),
          ((1.5, -0.4, 0.8), "#dfe9ff", 0.38),
          ((0.3, 1.2, -1.0), "#ffffff", 0.55),
          ((0.0, -1.2, 0.6), "#eef3ff", 0.22),
          ((0.0, 0.0, 1.0), "#ffffff", 0.15)]


LIGHTS_GLASS = [((-1.0, 1.3, 1.2), "#fff4e6", 0.95),
                ((0.3, 1.2, -1.0), "#ffffff", 0.45),
                ((0.0, 0.0, 1.0), "#ffffff", 0.35)]


def plotter(g, px, lights=None):
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
    pl.set_background("white")
    c, u, w = cam_axes()
    pl.camera.position = tuple(4 * g * c); pl.camera.focal_point = (0, 0, 0); pl.camera.up = tuple(w)
    pl.camera.enable_parallel_projection(); pl.camera.parallel_scale = SCALE * g
    pl.camera.clipping_range = (2 * g, 6 * g)
    for pos, col, it in lights or LIGHTS:
        pl.add_light(pv.Light(position=pos, focal_point=(0, 0, 0), color=col, intensity=it, light_type="camera light"))
    return pl


def finish(pl, path, ssaa=True, ssao=None):
    if ssao: pl.enable_ssao(radius=ssao[0], bias=ssao[1], kernel_size=128, blur=True)
    pl.enable_depth_peeling(number_of_peels=30, occlusion_ratio=0.0)
    if ssaa: pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(str(path), return_img=True)
    pl.close()
    return img


def fresnel_rgba(mesh, rgb, a0, a1, p=2.5):
    """Per-vertex colour and opacity: opacity grows towards the silhouette (glass seen at grazing angles)."""
    from matplotlib.colors import to_rgb
    c, _, _ = cam_axes(); n = mesh.point_data["Normals"]; f = 1 - np.abs(n @ c)
    a = a0 + (a1 - a0) * f ** p
    rgba = np.c_[np.tile(to_rgb(rgb), (mesh.n_points, 1)), a]
    return (255 * np.clip(rgba, 0, 1)).astype(np.uint8)


def tube(P, r, n_sides=16):
    import pyvista as pv
    return pv.lines_from_points(P).tube(radius=r, n_sides=n_sides, capping=True)


def pieces(P):
    """Split a NaN-separated polyline into its finite pieces (at least 2 points each)."""
    out = []
    for Q in np.split(P, np.where(~np.isfinite(P).all(1))[0]):
        Q = Q[np.isfinite(Q).all(1)]
        if len(Q) > 1: out.append(Q)
    return out


def dashed(P, r, dash):
    """Alternating dark and light tube segments of arc length `dash` (a dashed line with its own halo)."""
    import pyvista as pv
    segs = [[], []]
    for Q in pieces(P):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        n = max(int(round(s[-1] / dash)), 1); edges = np.linspace(0, s[-1], n + 1)
        for k in range(n):
            t = np.linspace(edges[k], edges[k + 1], 10); seg = np.stack([np.interp(t, s, Q[:, i]) for i in range(3)], -1)
            segs[k % 2].append(tube(seg, r, 12))
    return [pv.merge(x) for x in segs if x]


def sphere_contour(F, P, A, level, rad, drop_cut=True, front_only=True):
    """Level curve {F = level} of a function on the sphere of radius rad, as a NaN-separated polyline (by default only"""
    import matplotlib.pyplot as plt
    f = plt.figure(); ax = f.add_subplot(); cs = ax.contour(A, P, np.where(np.isfinite(F), F, 1e6), levels=[level]); plt.close(f)
    out = []
    for seg in cs.allsegs[0]:
        X = rad * dirs(seg[:, 1], seg[:, 0])
        if drop_cut: X[in_cut(X)] = np.nan
        if front_only: X[X @ cam_axes()[0] < 0] = np.nan
        out += [X, np.full((1, 3), np.nan)]
    return np.concatenate(out)


def plane_curves(rad, nt=181):
    """The curves r = rad(d) on the three quarter-planes bounding the cut octant (NaN-separated)."""
    segs = []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]; t = np.linspace(0, np.pi / 2, nt)
        d = np.zeros((nt, 3)); d[:, i] = CUT[i] * np.cos(t); d[:, j] = CUT[j] * np.sin(t)
        segs += [rad(d)[:, None] * d, np.full((1, 3), np.nan)]
    return np.concatenate(segs)


def interp_sphere(F, P, A, X):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((P[:, 0], A[0]), np.where(np.isfinite(F), F, 1e6))
    n = X / np.linalg.norm(X, axis=1, keepdims=True)
    return f(np.stack([np.arccos(np.clip(n[:, 0], -1, 1)), np.arctan2(n[:, 2], n[:, 1])], -1))


def radial_mesh(rad, f=2, drop=None):
    P, A = fine_grid(f); d = dirs(P, A)
    R = rad(d.reshape(-1, 3)).reshape(P.shape)
    return outward(grid_mesh(R[..., None] * d, drop=None if drop is None else drop(cell_dirs(P, A))))


def fail_sample(D, n):
    zf = D["z_fail"]; return zf[np.random.default_rng(0).choice(len(zf), n, replace=False)]


def render_a(D, path, px, opt):
    import pyvista as pv
    g = float(D["gamma"]); pl = plotter(g, px, LIGHTS_GLASS)
    ball = radial_mesh(M.Radius(D["r_V0"], D["pol"], D["az"]))
    pl.add_mesh(ball, scalars=fresnel_rgba(ball, opt["ball"], opt["ball_a0"], opt["ball_a1"]), rgba=True, smooth_shading=True,
                ambient=0.3, diffuse=0.7, specular=opt.get("ball_spec", 0.7), specular_power=opt.get("ball_sp", 90),
                culling=opt.get("ball_cull"))
    if opt.get("inner"):
        pl.add_mesh(ball.copy(), color=opt["inner"][0], opacity=opt["inner"][1], smooth_shading=True, culling="front",
                    ambient=0.6, diffuse=0.45, specular=0.2, specular_power=20)
    if opt.get("circles"):
        t = np.linspace(0, 2 * np.pi, 361)
        for k in range(3):
            i, j = [x for x in range(3) if x != k]; Q = np.zeros((361, 3)); Q[:, i] = np.cos(t); Q[:, j] = np.sin(t)
            c, _, _ = cam_axes(); Q = np.roll(Q, -int(np.argmin(Q @ c)), 0); Q = Q[Q @ c > 0.02]
            pl.add_mesh(tube(1.001 * g * Q, opt["circles"][0] * g), color=opt["circles"][1], opacity=opt["circles"][2],
                        smooth_shading=True, ambient=0.6)
    sph = pv.Sphere(radius=opt["fail_r"] * g, theta_resolution=16, phi_resolution=16)
    pl.add_mesh(pv.PolyData(fail_sample(D, opt["nfail"])).glyph(geom=sph, scale=False, orient=False), color=opt["fail"],
                smooth_shading=True, ambient=0.25, diffuse=0.8, specular=0.7, specular_power=50)
    return finish(pl, path)


def render_b(D, path, px, opt):
    import pyvista as pv
    g = float(D["gamma"]); pl = plotter(g, px)
    rV = M.Radius(D["r_Vf"], D["pol_F"], D["az_F"]); rU = lambda d: r_base(d, D)
    rO = M.Radius(D["r_O"], D["pol_O"], D["az_O"])
    cand = radial_mesh(rV, drop=in_cut)
    pl.add_mesh(cand, color=opt["cand"], opacity=opt["cand_op"], smooth_shading=True, culling=opt.get("cand_cull"),
                ambient=0.33, diffuse=0.72, specular=0.5, specular_power=30)
    for f_ in quarter_plane_faces(rU, rV):
        pl.add_mesh(f_, color=opt["cut"], opacity=1.0, ambient=0.72, diffuse=0.35, specular=0.05)
    base = radial_mesh(rU)
    pl.add_mesh(base, color=opt["base"], smooth_shading=True, ambient=0.28, diffuse=0.75, specular=0.4, specular_power=25)
    Ps, As = D["pol_S"], D["az_S"]
    cap = grid_mesh(1.003 * g * dirs(Ps, As), drop=in_cut(cell_dirs(Ps, As)))
    cap.point_data["sV"] = interp_sphere(D["sV"], Ps, As, cap.points)
    cap = outward(cap.clip_scalar(scalars="sV", value=float(D["rho"]), invert=False).extract_surface(algorithm="dataset_surface"))
    cap.point_data.pop("sV", None)
    pl.add_mesh(cap, scalars=fresnel_rgba(cap, opt["excl"], opt["excl_a0"], opt["excl_a1"]), rgba=True, smooth_shading=True,
                ambient=0.45, diffuse=0.6, specular=0.45, specular_power=50, culling="back")
    ew = opt["edge_r"] * g; ec = opt["edge"]
    for Q in pieces(sphere_contour(D["sV"], Ps, As, float(D["rho"]), 1.003 * g)) + pieces(plane_curves(rV)):
        pl.add_mesh(tube(Q, ew), color=ec, smooth_shading=True, ambient=0.5)
    for k in range(3):
        e_ = np.zeros(3); e_[k] = CUT[k]; pl.add_mesh(tube(np.outer([0, 1], rV(e_[None])[0] * e_), ew), color=ec, ambient=0.5)
    for Q in pieces(plane_curves(rU)):
        pl.add_mesh(tube(Q, 0.7 * ew), color=opt["base_edge"], smooth_shading=True, ambient=0.5)
    orim = sphere_contour(D["sO"], D["pol_SO"], D["az_SO"], float(D["rho_or"]), 1.006 * g)
    oc = plane_curves(rO); oc[np.linalg.norm(oc, axis=1) > 0.995 * g] = np.nan
    oc = oc + 0.006 * g * np.array([CUT[k] * (np.abs(oc[:, k]) < 1e-9) for k in range(3)]).T
    dk, lt = dashed(np.concatenate([orim, np.full((1, 3), np.nan), oc]), opt["orc_r"] * g, opt["dash"] * g)
    pl.add_mesh(dk, color=opt["orc"], smooth_shading=True, ambient=0.5, specular=0.3)
    pl.add_mesh(lt, color="white", smooth_shading=True, ambient=0.7)
    return finish(pl, path, ssao=opt.get("ssao"))


OPT = dict(ball=C_BALL, ball_a0=0.14, ball_a1=0.95, ball_cull="back", inner=(C_BALL_IN, 0.42), fail=C_FAIL, fail_r=0.012, nfail=4000,
           circles=None, cand=C_CAND, cand_op=1.0, cut=C_CAND_CUT, base=C_BASE, base_edge="#7a4a00",
           excl=C_EXCL, excl_a0=0.78, excl_a1=0.93, edge="#0b2a4a", edge_r=0.005, orc="#111111", orc_r=0.011, dash=0.07, ssao=(0.6, 0.02))


W, H = 6.3, 2.25
FS = 1.3
RECT_A = [0.03, 0.035, 0.29, 0.812]; RECT_B = [0.355, 0.035, 0.29, 0.812]; RECT_C = [0.80, 0.17, 0.19, 0.64]


def anchors(D):
    """3D anchor points of the leader lines, each on a visible part of its object."""
    g = float(D["gamma"]); c, u, w = cam_axes()
    rV = M.Radius(D["r_Vf"], D["pol_F"], D["az_F"])
    zs = fail_sample(D, OPT["nfail"]); cap = M.cap_direction(D)
    main = zs[(zs / np.linalg.norm(zs, axis=1, keepdims=True)) @ cap > 0.8]
    front = main[main @ c >= np.quantile(main @ c, 0.8)]
    ctr = np.median(main @ np.c_[u, w], axis=0); pF = front[np.argmin(np.sum((front @ np.c_[u, w] - ctr) ** 2, 1))]
    dB = np.array(CUT, float) / np.sqrt(3); pB = r_base(dB[None], D)[0] * dB
    dC = 0.45 * u - 0.75 * w + 0.5 * c; dC /= np.linalg.norm(dC); pC = rV(dC[None])[0] * dC
    orim = np.concatenate(pieces(sphere_contour(D["sO"], D["pol_SO"], D["az_SO"], float(D["rho_or"]), g)))
    orim = orim[orim @ c > 0.1 * g]; pO = orim[np.argmax(orim @ u)]
    P_, A_ = D["pol_S"], D["az_S"]; dS = dirs(P_, A_)
    m = (D["sV"] > float(D["rho"])) & (dS @ c > 0.2) & ~in_cut(dS); pts = g * dS[m]
    pX = pts[np.argmin((pts @ u - np.median(pts @ u)) ** 2 + (pts @ w - np.median(pts @ w)) ** 2)]
    return dict(fail=pF, base=pB, cand=pC, orc=pO, excl=pX)


def compose(D, img_a, img_b, stem, lab_off=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    M.style(); g = float(D["gamma"])
    import matplotlib as mpl
    for k_ in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"): mpl.rcParams[k_] *= FS
    fig = plt.figure(figsize=(W, H))
    ov = fig.add_axes([0, 0, 1, 1], zorder=10); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
    def place(rect, img):
        ax = fig.add_axes(rect); ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0), interpolation="lanczos")
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis("off"); return ax
    axa = place(RECT_A, img_a); axb = place(RECT_B, img_b)
    def fxy(rect, p, px):
        q = to_px(p, g, px)[0]; return rect[0] + q[0] / px * rect[2], rect[1] + (1 - q[1] / px) * rect[3]
    for rect, img in ((RECT_A, img_a), (RECT_B, img_b)):
        px = img.shape[0]; cx, cy = fxy(rect, np.zeros(3), px); r = g / (SCALE * g) * 0.5
        t = np.linspace(0, 2 * np.pi, 400)
        ov.plot(cx + r * rect[2] * np.cos(t), cy + r * rect[3] * np.sin(t), color=SIL, lw=0.45, zorder=1)
    an = anchors(D)
    off = dict(fail=(0.06, 0.075, "left"), excl=None, orc=(0.05, 0.07, "left"), base=None, cand=(0.05, -0.05, "left"))
    LAB_XY = dict(excl=(0.42, 0.875, "right"), base=(0.385, 0.088, "right"))
    if lab_off: off.update(lab_off)
    names = dict(fail="fails", excl="excluded", orc="oracle", base="$V_P$", cand="candidate")
    for k, o in off.items():
        rect, img = (RECT_A, img_a) if k == "fail" else (RECT_B, img_b)
        x, y = fxy(rect, an[k], img.shape[0])
        tx, ty, ha = (x + o[0], y + o[1], o[2]) if o else LAB_XY[k]
        ov.annotate(names[k], xy=(x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=8 * FS, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1.5, shrinkB=0.0))
        ov.plot([x], [y], "o", ms=1.6, color=INK, zorder=12)
    c, u, w = cam_axes(); o = np.array([0.07, 0.74]); L = 0.028
    for k, lbl in enumerate(["$z_1$", "$z_2$", "$z_3$"]):
        e_ = np.zeros(3); e_[k] = 1; v = np.array([e_ @ u, (e_ @ w) * W / H]) * L
        ov.plot([o[0], o[0] + v[0]], [o[1], o[1] + v[1]], color=INK2, lw=0.6, solid_capstyle="round")
        ov.text(o[0] + 1.55 * v[0], o[1] + 1.55 * v[1], lbl, fontsize=8 * FS, ha="center", va="center", color=INK)
    rows = loops55_pb()
    axc = fig.add_axes(RECT_C)
    v = np.array([r[1] / r[4] for r in rows]) * 100; vd = np.array([r[2] / r[4] for r in rows]) * 100
    vp = np.array([r[3] / r[4] for r in rows]) * 100; ub = np.maximum(vd, vp)
    o_ = np.argsort(ub, kind="stable"); x = np.arange(len(rows))
    ex = [k for k, j in enumerate(o_) if rows[j][0] == M.FID][0]
    axc.axvspan(ex - 0.5, ex + 0.5, color=BAND, lw=0, zorder=0)
    for k, j in enumerate(o_):
        axc.plot([k, k], [min(vd[j], vp[j]), v[j]], color=STEM, lw=1.6, zorder=1, solid_capstyle="round")
    axc.axhline(100, color=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=0.5)
    axc.scatter(x, vd[o_], s=9, marker="v", color=C_VD, edgecolor="white", lw=0.3, zorder=2, label="$V_d$")
    axc.scatter(x, vp[o_], s=9, color=C_BASE, edgecolor="white", lw=0.3, zorder=2.5, label="$V_P$")
    axc.scatter(x, v[o_], s=9, color=C_CAND, edgecolor="white", lw=0.3, zorder=3, label="$V$")
    axc.annotate("loop of (b)", xy=(ex + 0.3, ub[o_][ex] - 4), xytext=(ex + 2.5, 47), ha="left", va="center", fontsize=7 * FS,
                 color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0.5, shrinkB=0.5))
    axc.set_ylim(0, 104); axc.set_yticks([0, 50, 100]); axc.set_xticks([]); axc.set_xlim(-1, len(rows))
    axc.set_ylabel(r"% of oracle region", labelpad=2); axc.set_xlabel("closed loop", labelpad=3)
    for s_ in ("top", "right"): axc.spines[s_].set_visible(False)
    h_, l_ = axc.get_legend_handles_labels(); axc.legend([h_[2], h_[0], h_[1]], [l_[2], l_[0], l_[1]], loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, handletextpad=0.0, columnspacing=0.25, borderaxespad=0.15, markerscale=1.2, fontsize=7 * FS)
    for x_, t in ((RECT_A[0] + RECT_A[2] / 2, "(a) Not recomputed"), (RECT_B[0] + RECT_B[2] / 2, "(b) Recomputed"),
                  (RECT_C[0] + RECT_C[2] / 2 - 0.03, "(c) All actuator errors")):
        fig.text(x_, 0.955, t, ha="center", va="top", fontsize=8 * FS)
    layout_check(fig, [fxy(r, np.zeros(3), i.shape[0]) + (0.5 / SCALE * r[2],) for r, i in ((RECT_A, img_a), (RECT_B, img_b))], axc)
    fig.savefig(f"{stem}.png", dpi=300); fig.savefig(f"{stem}.pdf", dpi=600)
    plt.close(fig)


def layout_check(fig, discs, axc):
    """Every text inside the figure, no two texts overlapping, no text over a ball (disc) or over the plot area of (c)."""
    from matplotlib.text import Text
    fig.canvas.draw(); r = fig.canvas.get_renderer(); Wd, Hd = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text(), Text.get_window_extent(t, r)) for t in texts]
    bad = []
    for name, b in boxes:
        m_ = 0.1 * fig.dpi
        if b.x0 < m_ or b.y0 < m_ or b.x1 > Wd - m_ or b.y1 > Hd - m_: bad.append(("within 0.1 in of the border", name))
        for cx, cy, rad in discs:
            px_, py_, pr = cx * Wd, cy * Hd, rad * Wd
            qx, qy = np.clip(px_, b.x0, b.x1), np.clip(py_, b.y0, b.y1)
            if (qx - px_) ** 2 + (qy - py_) ** 2 < pr ** 2: bad.append(("over a ball", name))
    pa = axc.get_window_extent(r)
    for name, b in boxes:
        if name in ("loop of (b)", "$V$", "$V_d$", "$V_P$") and b.x0 > pa.x0: continue
        if b.overlaps(pa) and not (b.y1 <= pa.y0 or b.x1 <= pa.x0): bad.append(("over (c)", name))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]): bad.append(("text overlap", boxes[i][0], boxes[j][0]))
    print("layout check:", "ok" if not bad else bad, f"({len(boxes)} texts)")
    return bad


SIL, BAND, STEM = "#9aa3ad", "#e9eef5", "#c9d9ea"


def load_img(path, n=1100):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize((n, n), Image.LANCZOS)) if im.size[0] != n else np.asarray(im)


if __name__ == "__main__":
    if sys.argv[1] == "checks": checks()
    elif sys.argv[1] in ("a", "b"):
        D = dict(np.load(M.CACHE)); opt = dict(OPT); opt.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
        (render_a if sys.argv[1] == "a" else render_b)(D, SCR / sys.argv[2], int(opt.get("px", 1000)), opt)
    elif sys.argv[1] == "render":
        stem = sys.argv[2] if len(sys.argv) > 2 else "fig1_new"
        checks(); D = dict(np.load(M.CACHE))
        render_a(D, SCR / "panel_a.png", PX, OPT); render_b(D, SCR / "panel_b.png", PX, OPT)
        compose(D, load_img(SCR / "panel_a.png"), load_img(SCR / "panel_b.png"), str(SCR / stem))
    elif sys.argv[1] == "compose":
        D = dict(np.load(M.CACHE)); compose(D, load_img(SCR / sys.argv[3]), load_img(SCR / sys.argv[4]), str(SCR / sys.argv[2]))
