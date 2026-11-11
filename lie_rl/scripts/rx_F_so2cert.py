"""A proof, by interval arithmetic, of the conditions of the reported region on SO(2).

Usage: JAX_ENABLE_X64=1 python scripts/rx_F_so2cert.py DIR SEED CERT_PICKLE EVAL_JSON OUT_JSON
"""
import json, os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
import jax.numpy as jp
from lie_rl import certificate as C, variants as Vr, anchor as AN

d, seed, cert_path, eval_path, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
N0 = int(os.environ.get("N0", str(2 ** 14))); MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "16")); CHUNK = 20000
EPS = np.finfo(float).eps; PAD = 1e-14; ABS_PAD = 1e-15
t0 = time.time()
cl = Vr.ClosedLoop("SO2", d, seed, "base"); amax = float(cl.amax)
assert json.load(open(eval_path))["variant"] == "base", "the interval model hard-codes the unperturbed SO(2) loop"
cert = pickle.load(open(cert_path, "rb")); cfg = C.CertConfig(**cert["cfg"]); gamma = cfg.radius; r_b = cfg.eps_ball
assert cert.get("anchor") is not None
P_m = float(np.asarray(cert["anchor"]["P"]).reshape(-1)[0]); r_a = float(cert["anchor"]["r_a"])
psi = [(np.asarray(w, float), np.asarray(b, float)) for w, b in cert["psi"]]
kappa = float(cfg.alpha_floor + np.log1p(np.exp(float(cert["alpha_raw"]))))
actor = [(np.asarray(w, float), np.asarray(b, float)) for w, b in cl.af]
bias = np.asarray(cl.bias, float).reshape(-1)
rho = json.load(open(eval_path))["fns"]["V"]["level"]


def pad(lo, hi):
    return lo - PAD * np.abs(lo) - ABS_PAD, hi + PAD * np.abs(hi) + ABS_PAD


def imul(a, b):
    p = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return pad(p.min(0), p.max(0))


def iadd(a, b): return pad(a[0] + b[0], a[1] + b[1])
def isub(a, b): return pad(a[0] - b[1], a[1] - b[0])
def iscale(a, k): return pad(np.minimum(a[0] * k, a[1] * k), np.maximum(a[0] * k, a[1] * k))


def isq(a):
    lo = np.where((a[0] <= 0) & (a[1] >= 0), 0.0, np.minimum(a[0] ** 2, a[1] ** 2)); return pad(lo, np.maximum(a[0] ** 2, a[1] ** 2))


def iaffine(a, W, b=None):
    """x @ W + b for an interval vector x (rows = cells), centre-radius form with the rounding error bound."""
    c = 0.5 * (a[0] + a[1]); r = 0.5 * (a[1] - a[0]); n = W.shape[0]; aW = np.abs(W)
    cc = c @ W + (0 if b is None else b)
    err = 2 * n * EPS * (np.abs(c) @ aW + (0 if b is None else np.abs(b))) + (r @ aW) * (2 * n * EPS)
    rr = r @ aW + err
    return pad(cc - rr, cc + rr)


def trig_range(lo, hi, f, crit):
    """exact range of f in {sin, cos} on [lo, hi] (critical points crit + k pi), padded."""
    vals = [f(lo), f(hi)]; mn = np.minimum(*vals); mx = np.maximum(*vals)
    for k in range(-3, 4):
        x = crit + k * np.pi; inside = (lo <= x) & (x <= hi); fx = f(x)
        mn = np.where(inside, np.minimum(mn, fx), mn); mx = np.where(inside, np.maximum(mx, fx), mx)
    return pad(mn - 2 * EPS, mx + 2 * EPS)


def icos(a): return trig_range(a[0], a[1], np.cos, 0.0)
def isin(a): return trig_range(a[0], a[1], np.sin, np.pi / 2)


def itanh(v, dv):
    tv = pad(np.tanh(v[0]), np.tanh(v[1]))
    t2 = np.stack([1 - np.tanh(v[0]) ** 2, 1 - np.tanh(v[1]) ** 2])
    z0 = np.clip(0.0, v[0], v[1]); dmax = 1 - np.tanh(z0) ** 2
    fac = pad(np.maximum(t2.min(0), 0.0), np.minimum(dmax, 1.0))
    return tv, imul(fac, dv)


def irelu(v, dv):
    fac = ((v[0] > 0).astype(float), (v[1] > 0).astype(float))
    return (np.maximum(v[0], 0), np.maximum(v[1], 0)), imul(fac, dv)


def mlp(x, dx, params, act, out_act=None):
    for W, b in params[:-1]:
        x, dx = act(iaffine(x, W, b), iaffine(dx, W))
    W, b = params[-1]; x, dx = iaffine(x, W, b), iaffine(dx, W)
    if out_act is not None: x, dx = out_act(x, dx)
    return x, dx


def col(a): return (a[0][:, None], a[1][:, None])


def mu_identity_interval():
    """an interval enclosing the exact mu(I): the actor at the observation of the identity, cos 0 = 1 and sin 0 = 0 exactly."""
    o = np.array([[1.0, 0.0, 0.0, 1.0]]); a, _ = mlp((o, o), (np.zeros_like(o), np.zeros_like(o)), actor, itanh, itanh)
    return a[0][0], a[1][0]


def F_int(z):
    """F(z) and F'(z) on the cells z = (lo, hi)."""
    c, s = icos(z), isin(z)
    obs = (np.stack([c[0], -s[1], s[0], c[0]], 1), np.stack([c[1], -s[0], s[1], c[1]], 1))
    dobs = (np.stack([-s[1], -c[1], c[0], -s[1]], 1), np.stack([-s[0], -c[0], c[1], -s[0]], 1))
    a, da = mlp(obs, dobs, actor, itanh, itanh)
    a = isub(a, BIAS_I)
    lo, hi = a[0][:, 0], a[1][:, 0]; dlo, dhi = da[0][:, 0], da[1][:, 0]
    clo, chi = np.clip(lo, -1, 1), np.clip(hi, -1, 1)
    inside = (lo > -1) & (hi < 1); outside = (hi < -1) | (lo > 1)
    fac = (np.where(inside, 1.0, 0.0), np.where(outside, 0.0, 1.0))
    dc = imul(fac, (dlo, dhi))
    Fv = iadd(z, iscale((clo, chi), amax)); dF = iadd((np.ones_like(lo), np.ones_like(lo)), iscale(dc, amax))
    return Fv, dF


BIAS_I = mu_identity_interval()
assert np.all(BIAS_I[0] - 1e-12 <= bias) and np.all(bias <= BIAS_I[1] + 1e-12), "stored bias far from mu(I)"
psi0 = np.zeros((1, 1)); h0 = psi0
for W, b in psi[:-1]: h0 = np.maximum(h0 @ W + b, 0)
psi0 = (h0 @ psi[-1][0] + psi[-1][1])[0]


def V_int(y):
    """V(y) and V'(y) on intervals y = (lo, hi)."""
    one = np.ones_like(y[0])
    ps, dps = mlp(col(y), (one[:, None], one[:, None]), psi, irelu)
    dif = isub(ps, (psi0, psi0))
    sq = isq(dif); dsq = imul(iscale(dif, 2.0), dps)
    S = (sq[0].sum(1), sq[1].sum(1)); dS = (dsq[0].sum(1), dsq[1].sum(1))
    S = pad(*S); dS = pad(*dS)
    l2 = iscale(isub((one, one), icos(y)), 4.0); dl2 = iscale(isin(y), 4.0)
    V0 = iadd(S, iscale(l2, kappa ** 2)); dV0 = iadd(dS, iscale(dl2, kappa ** 2))
    y2 = isq(y); q = iscale(y2, P_m); dq = iscale(y, 2 * P_m)
    ra2 = r_a ** 2
    s = pad(y2[0] / (y2[0] + ra2), y2[1] / (y2[1] + ra2))
    k = pad(ra2 / (y2[1] + ra2) ** 2, ra2 / (y2[0] + ra2) ** 2)
    ds = imul(iscale(y, 2.0), k)
    D = isub(V0, q); dD = isub(dV0, dq)
    V = iadd(q, imul(s, D)); dV = iadd(dq, iadd(imul(ds, D), imul(s, dD)))
    return V, dV, l2


def stats(lo, hi):
    out = {k: [] for k in ("Vlo", "Vhi", "VFhi", "Flo", "Fhi", "l2lo", "l2hi", "gU", "dFabs")}
    for i in range(0, len(lo), CHUNK):
        z = (lo[i:i + CHUNK], hi[i:i + CHUNK]); zc = 0.5 * (z[0] + z[1]); h = 0.5 * (z[1] - z[0])
        Fv, dF = F_int(z); Vz, dVz, l2 = V_int(z); VF, dVF, _ = V_int(Fv)
        Fc, _ = F_int((zc, zc)); Vc, _, _ = V_int((zc, zc)); VFc, _, _ = V_int(Fc)
        gp = isub(imul(dVF, dF), dVz)
        g_centered = (VFc[1] - Vc[0]) + np.maximum(np.abs(gp[0]), np.abs(gp[1])) * h * (1 + 1e-12)
        out["gU"].append(np.minimum(g_centered, VF[1] - Vz[0]))
        for k_, v_ in (("Vlo", Vz[0]), ("Vhi", Vz[1]), ("VFhi", VF[1]), ("Flo", Fv[0]), ("Fhi", Fv[1]), ("l2lo", l2[0]), ("l2hi", l2[1]),
                       ("dFabs", np.maximum(np.abs(dF[0]), np.abs(dF[1])))):
            out[k_].append(v_)
    return {k: np.concatenate(v) for k, v in out.items()}


def failing(S, lvl):
    meet = S["Vlo"] <= lvl
    dec = meet & (S["l2hi"] >= r_b ** 2) & (S["gU"] >= 0)
    cont = meet & ((S["Flo"] < -gamma) | (S["Fhi"] > gamma))
    ent = meet & (S["l2lo"] < r_b ** 2) & (S["VFhi"] > lvl)
    return dec, cont, ent


edges = np.linspace(-gamma, gamma, N0 + 1); lo, hi = edges[:-1], edges[1:]
keep = {k: [] for k in ("lo", "hi")}; keepS = []; depth_hist = []
for depth in range(MAX_DEPTH + 1):
    S = stats(lo, hi); dec, cont, ent = failing(S, rho); bad = dec | cont | ent
    depth_hist.append(dict(depth=depth, cells=int(len(lo)), failing=int(bad.sum()), dec=int(dec.sum()), cont=int(cont.sum()), entry=int(ent.sum())))
    print(json.dumps(depth_hist[-1]), flush=True)
    ok = ~bad if depth < MAX_DEPTH else np.ones_like(bad)
    keep["lo"].append(lo[ok]); keep["hi"].append(hi[ok]); keepS.append({k: v[ok] for k, v in S.items()})
    if depth == MAX_DEPTH or not bad.any(): break
    mid = 0.5 * (lo[bad] + hi[bad]); lo, hi = np.concatenate([lo[bad], mid]), np.concatenate([mid, hi[bad]])
LO, HI = np.concatenate(keep["lo"]), np.concatenate(keep["hi"]); o = np.argsort(LO); LO, HI = LO[o], HI[o]
S = {k: np.concatenate([s[k] for s in keepS])[o] for k in keepS[0]}

lvl = rho; it = 0
while True:
    dec, cont, ent = failing(S, lvl); bad = dec | cont | ent
    if not bad.any() or it > 200: break
    lvl = (1 - 1e-12) * float(S["Vlo"][bad].min()); it += 1
final_bad = bool((lambda t: (t[0] | t[1] | t[2]).any())(failing(S, lvl)))


def merged(mask):
    iv = []
    for a, b in zip(LO[mask], HI[mask]):
        if iv and abs(a - iv[-1][1]) < 1e-15: iv[-1][1] = b
        else: iv.append([float(a), float(b)])
    return iv


dec0, cont0, ent0 = failing(S, rho)
delta = float(np.arccos(1 - r_b ** 2 / 4)); dl = np.linspace(-delta * (1 + 1e-9), delta * (1 + 1e-9), 4097)
_, dFl = F_int((dl[:-1], dl[1:])); q_loc = float(np.max(np.maximum(np.abs(dFl[0]), np.abs(dFl[1]))))
eps_loc = delta
while eps_loc > 1e-12:
    e = np.linspace(-eps_loc * (1 + 1e-9), eps_loc * (1 + 1e-9), 257); _, dFe = F_int((e[:-1], e[1:]))
    q_eps = float(np.max(np.maximum(np.abs(dFe[0]), np.abs(dFe[1]))))
    if q_eps < 1: break
    eps_loc /= 2
ratio_ok, ratio_max, ratio_cells = False, None, 0
if eps_loc > 1e-12:
    g_ = np.linspace(eps_loc, delta * (1 + 1e-9), 4097); a, b = np.concatenate([g_[:-1], -g_[1:]]), np.concatenate([g_[1:], -g_[:-1]])
    worst = 0.0
    for _ in range(30):
        Fr, _ = F_int((a, b)); inf_abs = np.minimum(np.abs(a), np.abs(b)); sup_F = np.maximum(np.abs(Fr[0]), np.abs(Fr[1]))
        ok = sup_F < inf_abs; ratio_cells += int(ok.sum()); worst = max(worst, float((sup_F[ok] / inf_abs[ok]).max()) if ok.any() else 0.0)
        a, b = a[~ok], b[~ok]
        if len(a) == 0: break
        mid = 0.5 * (a + b); a, b = np.concatenate([a, mid]), np.concatenate([mid, b])
    ratio_ok, ratio_max = len(a) == 0, worst
local_ok = bool(q_loc < 1 or ratio_ok)
w = HI - LO
m_lo = float(w[S["Vhi"] <= lvl].sum()); a, b = LO[(S["Vlo"] <= lvl) & (S["Vhi"] > lvl)], HI[(S["Vlo"] <= lvl) & (S["Vhi"] > lvl)]
for _ in range(int(os.environ.get("MEASURE_SPLITS", "20"))):
    if len(a) == 0: break
    mid = 0.5 * (a + b); a, b = np.concatenate([a, mid]), np.concatenate([mid, b])
    Vv = [V_int((a[i:i + CHUNK], b[i:i + CHUNK]))[0] for i in range(0, len(a), CHUNK)]
    vlo, vhi = np.concatenate([v[0] for v in Vv]), np.concatenate([v[1] for v in Vv])
    m_lo += float((b - a)[vhi <= lvl].sum()); st = (vlo <= lvl) & (vhi > lvl); a, b = a[st], b[st]
m_hi = m_lo + float((b - a).sum())
rec = dict(group="SO2", dir=d, seed=seed, cert=str(cert_path), eval=str(eval_path), gamma=gamma, r_b=r_b, alpha_max=amax, r_a=r_a,
           status="interval arithmetic in float64 with rounding-error bounds folded into the radii (not directed rounding)",
           level_reported=rho, cells=int(len(LO)), smallest_cell=float(w.min()), depth_history=depth_hist,
           proved_at_reported_level=bool(not (dec0 | cont0 | ent0).any()),
           unproved_at_reported_level=dict(decrease=merged(dec0), containment=merged(cont0), entry=merged(ent0)),
           level_proved=None if final_bad else lvl, level_proved_over_reported=None if final_bad else (lvl / rho if rho else None),
           cells_failing_at_level_proved=final_bad,
           region_mu0_reported=json.load(open(eval_path))["fns"]["V"]["region_mu0"],
           region_mu0_proved_lower=None if final_bad else m_lo / (2 * np.pi), region_mu0_proved_upper=None if final_bad else m_hi / (2 * np.pi),
           min_decrease_margin=float(-S["gU"][(S["Vlo"] <= lvl) & (S["l2hi"] >= r_b ** 2)].max()) if ((S["Vlo"] <= lvl) & (S["l2hi"] >= r_b ** 2)).any() else None,
           local_delta=delta, local_sup_abs_dF=q_loc, local_contraction_proved=bool(q_loc < 1),
           local_ratio_eps=eps_loc, local_ratio_sup_abs_dF_on_eps=q_eps, local_ratio_proved=bool(ratio_ok),
           local_ratio_max_bound=ratio_max, local_ratio_cells=ratio_cells, local_step_proved=local_ok,
           bias_interval=[float(BIAS_I[0][0]), float(BIAS_I[1][0])], bias_stored=float(bias[0]),
           containment_margin=float(np.min(np.minimum(gamma - S["Fhi"], S["Flo"] + gamma)[S["Vlo"] <= lvl])),
           local_note="mu(I) enters as an interval enclosing its exact value, so F(0) = 0 holds exactly for the corrected policy u = mu - mu(I)",
           proved=bool(local_ok and lvl > 0 and not final_bad), wall_s=time.time() - t0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k not in ("depth_history",)}))
