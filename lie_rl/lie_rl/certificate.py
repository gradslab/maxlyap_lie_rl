"""Learned Zubov certificate on a general matrix Lie group."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax

from . import env as E
from .nets import init_mlp, mlp_apply


class CertParams(NamedTuple):
    psi: list
    alpha_raw: jax.Array
    eps_raw: jax.Array


class CertConfig(NamedTuple):
    m: int = 3
    hidden: tuple = (64, 64)
    feat_out: int = 16
    alpha_floor: float = 0.05
    alpha_init: float = 0.3
    eps_init: float = 0.2
    radius: float = 2.7489
    eps_ball: float = 0.1
    zeta_train: float = 0.1
    alpha_max: float = 0.3142
    iters: int = 6000
    batch: int = 4096
    lr: float = 1e-3
    buffer_size: int = 32768
    resample_every: int = 200
    pgd_every: int = 10
    pgd_batch: int = 1024
    pgd_steps: int = 30
    ce_batch: int = 1024
    mine_batch: int = 262_144
    mine_keep: int = 2048
    ce_buffer_max: int = 16384
    w_ce: float = 5.0
    lam_lr: float = 0.02
    lam_max: float = 50.0
    w_data: float = 0.2
    eps_l2: float = 1e-3
    spec_target: float = 2.0
    w_spec: float = 1e-3
    w_shell: float = 0.3
    shell_batch: int = 2048
    mine_scoped: bool = True
    mine_exit: bool = False
    w_exit: float = 1.0
    exit_margin: float = 1.2
    rho_refresh_every: int = 200
    shell_probe: int = 16384
    rollout_len: int = 300
    seed: int = 0
    cost_power: int = 1
    data_mode: str = "mse"
    zubov_mode: str = "W"
    dec_mode: str = "abs"
    zeta_rel: float = 0.05
    eps_fixed: float = 0.0
    use_bnd: bool = True
    input_mode: str = "chart"
    obs_dim: int = 9
    exclude_center: tuple = ()
    exclude_chord: float = 0.0
    radial_frac: float = 0.0


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-6))))


def init_cert(key, cfg: CertConfig) -> CertParams:
    d_in = cfg.m if cfg.input_mode == "chart" else cfg.obs_dim
    psi = init_mlp(key, [d_in, *cfg.hidden, cfg.feat_out], scale="lecun")
    return CertParams(psi=psi,
                      alpha_raw=jp.asarray(_softplus_inv(cfg.alpha_init - cfg.alpha_floor)),
                      eps_raw=jp.asarray(_softplus_inv(cfg.eps_init)))


def alpha_of(p, cfg):
    return cfg.alpha_floor + jax.nn.softplus(p.alpha_raw)


def epsp_of(p):
    return jax.nn.softplus(p.eps_raw)


def sample_ball(key, n, m, radius, r_min=0.0):
    k1, k2 = jax.random.split(key)
    d = jax.random.normal(k1, (n, m))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    u = jax.random.uniform(k2, (n, 1))
    r = (r_min ** m + (radius ** m - r_min ** m) * u) ** (1.0 / m)
    return d * r


def sample_ball_radial(key, n, m, radius, r_min=0.0):
    """direction uniform on the sphere, radius uniform on [r_min, radius]: covers the neighbourhood"""
    k1, k2 = jax.random.split(key)
    d = jax.random.normal(k1, (n, m))
    d = d / jp.linalg.norm(d, axis=-1, keepdims=True)
    r = r_min + (radius - r_min) * jax.random.uniform(k2, (n, 1))
    return d * r


def sample_mixed(key, n, m, radius, radial_frac):
    """volume-uniform samples with a fraction radial_frac replaced by radius-uniform ones."""
    if radial_frac <= 0: return sample_ball(key, n, m, radius)
    k1, k2 = jax.random.split(key); n_r = int(round(radial_frac * n))
    return jp.concatenate([sample_ball(k1, n - n_r, m, radius), sample_ball_radial(k2, n_r, m, radius)], 0)


def make_ell(G):
    """ell(z) = || vec(Exp z) - vec(e) ||, the chord distance in the chart."""
    ref = G.obs(G.identity(1))

    def ell(z):
        return jp.linalg.norm(G.obs(G.Exp(z)) - ref, axis=-1)
    ell.obs = lambda z: G.obs(G.Exp(z))
    return ell


def _inp(cfg, ell, z):
    return ell.obs(z) if cfg.input_mode == "group" else z


def _hinge_mask(cfg, ell, z, lz):
    mask = (lz >= cfg.eps_ball)
    if cfg.input_mode == "group" and cfg.exclude_chord > 0 and len(cfg.exclude_center) > 0:
        centers = jp.asarray(cfg.exclude_center, dtype=z.dtype).reshape(-1, z.shape[-1])
        C_obs = ell.obs(centers)
        dmin = jp.min(jp.linalg.norm(ell.obs(z)[:, None, :] - C_obs[None, :, :], axis=-1), axis=1)
        mask = mask & (dmin >= cfg.exclude_chord)
    return mask.astype(z.dtype)


def cost_of(cfg, ell, z):
    """running cost c(z) = ell(z) ** cost_power."""
    return ell(z) ** cfg.cost_power


def V_chart(p, cfg, ell, z):
    a = alpha_of(p, cfg)
    feats = mlp_apply(p.psi, _inp(cfg, ell, z), jax.nn.relu)
    feats0 = mlp_apply(p.psi, _inp(cfg, ell, jp.zeros((1, cfg.m))), jax.nn.relu)
    return jp.sum((feats - feats0) ** 2, axis=-1) + a ** 2 * ell(z) ** 2


def W_chart(p, cfg, ell, z):
    return 1.0 - jp.exp(-epsp_of(p) * V_chart(p, cfg, ell, z))


class LossPieces(NamedTuple):
    dec: jax.Array
    ce: jax.Array
    zubov: jax.Array
    bnd: jax.Array
    data: jax.Array
    reg: jax.Array
    exit: jax.Array


def loss_pieces(p, cfg, ell, z, zT, z_data, vm_data, z_ce, zT_ce, z_exit=None):
    V0 = V_chart(p, cfg, ell, z)
    V1 = V_chart(p, cfg, ell, zT)
    epsp = epsp_of(p)
    W0 = 1.0 - jp.exp(-epsp * V0)
    W1 = 1.0 - jp.exp(-epsp * V1)
    lz = ell(z)
    cz = cost_of(cfg, ell, z)

    mask = _hinge_mask(cfg, ell, z, lz)
    if cfg.dec_mode == "rel":
        dec = jp.sum(jax.nn.relu(V1 - (1.0 - cfg.zeta_rel) * V0) * mask) / jp.clip(mask.sum(), 1.0)
    else:
        dec = jp.sum(jax.nn.relu(V1 - V0 + cfg.zeta_train) * mask) / jp.clip(mask.sum(), 1.0)

    if cfg.zubov_mode == "V":
        zubov = jp.mean((V1 - V0 + cz) ** 2)
    else:
        zubov = jp.mean((W1 - 1.0 + (1.0 - W0) * jp.exp(epsp * cz)) ** 2)

    exits = (jp.linalg.norm(zT, axis=-1) > cfg.radius).astype(z.dtype)
    bnd = jp.sum(jax.nn.relu(1.0 - W0) ** 2 * exits) / jp.clip(exits.sum(), 1.0)

    W_target = 1.0 - jp.exp(-epsp * vm_data)
    if cfg.data_mode == "near":
        wts = jax.lax.stop_gradient(1.0 - W_target)
        data = jp.sum(wts * (W_chart(p, cfg, ell, z_data) - W_target) ** 2) / jp.clip(jp.sum(wts), 1e-9)
    elif cfg.data_mode == "log":
        data = jp.mean((jp.log1p(V_chart(p, cfg, ell, z_data)) - jp.log1p(vm_data)) ** 2)
    else:
        data = jp.mean((W_chart(p, cfg, ell, z_data) - W_target) ** 2)

    if z_ce is not None and z_ce.shape[0] > 0:
        Vc0 = V_chart(p, cfg, ell, z_ce)
        Vc1 = V_chart(p, cfg, ell, zT_ce)
        mc = _hinge_mask(cfg, ell, z_ce, ell(z_ce))
        if cfg.dec_mode == "rel":
            ce = jp.sum(jax.nn.relu(Vc1 - (1.0 - cfg.zeta_rel) * Vc0) * mc) / jp.clip(mc.sum(), 1.0)
        else:
            ce = jp.sum(jax.nn.relu(Vc1 - Vc0 + cfg.zeta_train) * mc) / jp.clip(mc.sum(), 1.0)
    else:
        ce = jp.zeros(())

    if z_exit is not None and z_exit.shape[0] > 0:
        target = cfg.exit_margin * jax.lax.stop_gradient(jp.max(V0)) + 1e-9
        exit_term = jp.mean(jax.nn.relu(1.0 - V_chart(p, cfg, ell, z_exit) / target) ** 2)
    else:
        exit_term = jp.zeros(())

    spec = sum(jax.nn.relu(jp.linalg.norm(w, ord=2) - cfg.spec_target) ** 2 for w, _ in p.psi)
    reg = cfg.eps_l2 * epsp ** 2 + cfg.w_spec * spec
    return LossPieces(dec, ce, zubov, bnd, data, reg, exit_term)


def make_vm(T_chart, ell, cfg):
    """Vm(z) = sum_{k>=0} ell(z_k) along the closed loop, the regression target."""
    def vm(z):
        def body(carry, _):
            zz, acc = carry
            acc = acc + cost_of(cfg, ell, zz)
            return (T_chart(zz), acc), None
        (_, acc), _ = jax.lax.scan(body, (z, jp.zeros(z.shape[0])), None,
                                   length=cfg.rollout_len)
        return acc
    return vm


def pgd_counterexamples(p, cfg, ell, T_chart, key):
    def gap(z):
        return V_chart(p, cfg, ell, T_chart(z)) - V_chart(p, cfg, ell, z)
    z = sample_ball(key, cfg.pgd_batch, cfg.m, cfg.radius)
    step = cfg.radius / cfg.pgd_steps

    def body(z, _):
        g = jax.grad(lambda zz: gap(zz).sum())(z)
        z = z + step * jp.sign(g)
        n = jp.linalg.norm(z, axis=-1, keepdims=True)
        return z * jp.minimum(1.0, cfg.radius / jp.clip(n, 1e-12)), None

    z, _ = jax.lax.scan(body, z, None, length=cfg.pgd_steps)
    return z, gap(z)


def train_certificate(T_chart, G, cfg: CertConfig, verbose=True):
    ell = make_ell(G)
    vm_fn = make_vm(T_chart, ell, cfg)
    key = jax.random.key(cfg.seed)
    key, k0, kd, kb = jax.random.split(key, 4)
    p = init_cert(k0, cfg)
    if cfg.eps_fixed > 0:
        p = p._replace(eps_raw=jp.asarray(_softplus_inv(cfg.eps_fixed)))
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(p)

    buf = sample_mixed(kb, cfg.buffer_size, cfg.m, cfg.radius, cfg.radial_frac)
    z_data = sample_ball(kd, 4096, cfg.m, cfg.radius)
    vm_data = vm_fn(z_data)
    ce = jp.zeros((0, cfg.m))
    ce_exit = jp.zeros((0, cfg.m))
    lam = {"dec": 1.0, "bnd": 1.0 if cfg.use_bnd else 0.0}
    rho_cur = jp.inf

    @jax.jit
    def step_fn(p, opt_state, zb, zb_ce, zb_exit, z_data_b, vm_b, z_shell, lam_dec, lam_bnd, w_z, w_exit):
        zT = T_chart(zb)
        zT_ce = T_chart(zb_ce)

        def total(pp):
            L = loss_pieces(pp, cfg, ell, zb, zT, z_data_b, vm_b, zb_ce, zT_ce, zb_exit)
            lam_data = jp.exp(-0.5 * (lam_dec + lam_bnd) / 10.0) * cfg.w_data
            V_sh = V_chart(pp, cfg, ell, z_shell)
            med = jax.lax.stop_gradient(jp.median(V_sh)) + 1e-9
            shell = jp.mean(jax.nn.relu(1.0 - V_sh / med) ** 2)
            return (lam_dec * L.dec + cfg.w_ce * L.ce + w_z * L.zubov
                    + lam_bnd * L.bnd + lam_data * L.data + w_exit * L.exit
                    + cfg.w_shell * shell + L.reg), (L, shell)

        (val, (L, shell)), g = jax.value_and_grad(total, has_aux=True)(p)
        if cfg.eps_fixed > 0:
            g = g._replace(eps_raw=jp.zeros_like(g.eps_raw))
        upd, opt_state = opt.update(g, opt_state)
        p = jax.tree.map(lambda a, b: a + b, p, upd)
        return p, opt_state, L, shell

    hist = []
    n_skipped = 0
    for it in range(cfg.iters):
        key, k1, k2, k3, km = jax.random.split(key, 5)
        if it % cfg.resample_every == 0 and it > 0:
            n_new = cfg.buffer_size // 4
            buf = jp.concatenate([buf[n_new:],
                                  sample_mixed(k1, n_new, cfg.m, cfg.radius, cfg.radial_frac)])
        if cfg.mine_scoped and it % cfg.rho_refresh_every == 0:
            key, ks = jax.random.split(key)
            z_sh = sample_ball(ks, cfg.shell_probe, cfg.m,
                               cfg.radius + cfg.alpha_max, r_min=cfg.radius)
            rho_cur = float(jp.min(V_chart(p, cfg, ell, z_sh)))
        if cfg.mine_keep > 0 and it % cfg.pgd_every == 0:
            z_ce, gaps = pgd_counterexamples(p, cfg, ell, T_chart, k3)
            hits = z_ce[gaps > -cfg.zeta_train]
            z_rnd = sample_ball(km, cfg.mine_batch, cfg.m, cfg.radius)
            zT_rnd = T_chart(z_rnd)
            gap_rnd = (V_chart(p, cfg, ell, zT_rnd)
                       - V_chart(p, cfg, ell, z_rnd))
            if cfg.mine_exit:
                leaves = (jp.linalg.norm(zT_rnd, axis=-1) > cfg.radius) & (ell(z_rnd) >= cfg.eps_ball)
                if bool(jp.any(leaves)):
                    ce_exit = jp.concatenate([ce_exit, z_rnd[leaves]])[-cfg.ce_buffer_max:]
            if cfg.mine_scoped:
                in_level = V_chart(p, cfg, ell, z_rnd) <= rho_cur
                gap_rnd = jp.where(in_level, gap_rnd, -jp.inf)
                if hits.shape[0]:
                    hits = hits[V_chart(p, cfg, ell, hits) <= rho_cur]
            worst = jp.argsort(gap_rnd)[-cfg.mine_keep:]
            mined = z_rnd[worst][gap_rnd[worst] > -cfg.zeta_train]
            hits = jp.concatenate([hits, mined])
            if hits.shape[0]:
                ce = jp.concatenate([ce, hits])[-cfg.ce_buffer_max:]

        idx = jax.random.randint(k2, (cfg.batch,), 0, buf.shape[0])
        zb = buf[idx]
        if ce.shape[0] >= cfg.ce_batch:
            jdx = jax.random.randint(k1, (cfg.ce_batch,), 0, ce.shape[0])
            zb_ce = ce[jdx]
        else:
            zb_ce = zb[:cfg.ce_batch]
        if ce_exit.shape[0] > 0:
            edx = jax.random.randint(k3, (cfg.ce_batch,), 0, ce_exit.shape[0])
            zb_exit = ce_exit[edx]; w_exit_eff = cfg.w_exit
        else:
            zb_exit = zb[:cfg.ce_batch]; w_exit_eff = 0.0
        ddx = jax.random.randint(k3, (1024,), 0, z_data.shape[0])
        w_z = min(1.0, it / max(1, int(0.2 * cfg.iters)))

        key, ksh = jax.random.split(key)
        z_shell = sample_ball(ksh, cfg.shell_batch, cfg.m,
                              cfg.radius + cfg.alpha_max, r_min=cfg.radius)
        p_prev, os_prev = p, opt_state
        p, opt_state, L, shell = step_fn(p, opt_state, zb, zb_ce, zb_exit, z_data[ddx],
                                         vm_data[ddx], z_shell,
                                         lam["dec"], lam["bnd"], w_z, w_exit_eff)
        if not all(np.isfinite(float(x)) for x in (*L, shell)):
            p, opt_state = p_prev, os_prev
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 100 == 0:
                print(f"    it {it:5d}  non-finite loss, step skipped ({n_skipped} so far)", flush=True)
            continue
        lam["dec"] = float(np.clip(lam["dec"] + cfg.lam_lr * float(L.dec), 0.0, cfg.lam_max))
        lam["bnd"] = float(np.clip(lam["bnd"] + cfg.lam_lr * float(L.bnd), 0.0, cfg.lam_max)) if cfg.use_bnd else 0.0
        if verbose and it % 500 == 0:
            print(f"    it {it:5d}  rho_cur {float(rho_cur):.4g}  "
                  f"dec {float(L.dec):.3e}  ce {float(L.ce):.3e}  "
                  f"zubov {float(L.zubov):.3e}  bnd {float(L.bnd):.3e}  "
                  f"data {float(L.data):.3e}  shell {float(shell):.3e}  "
                  f"exit {float(L.exit):.3e}  |ce| {ce.shape[0]}  |exit| {ce_exit.shape[0]}", flush=True)
        hist.append(float(L.dec))
    return p, hist
