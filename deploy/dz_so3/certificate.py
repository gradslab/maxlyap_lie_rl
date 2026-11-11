"""Certificate stack: V_phi architecture, discrete Zubov losses, PGD, training."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax

from . import so3
from .nets import init_mlp, mlp_apply


class CertParams(NamedTuple):
    psi: list
    alpha_raw: jax.Array
    eps_raw: jax.Array


class CertConfig(NamedTuple):
    hidden: tuple = (64, 64)
    feat_out: int = 16
    alpha_floor: float = 0.05
    alpha_init: float = 0.3
    eps_init: float = 0.2
    radius: float = float(np.pi - np.pi / 8)
    eps_ball: float = 0.1
    zeta_train: float = 0.1
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
    primal_dual: bool = True
    lam_lr: float = 0.02
    lam_max: float = 50.0
    w_data: float = 0.2
    eps_l2: float = 1e-3
    spec_target: float = 2.0
    w_spec: float = 1e-3
    quad_only: bool = False
    alpha_max: float = float(np.pi * 0.1)
    mine_scoped: bool = False
    rho_refresh_every: int = 200
    w_shell: float = 0.0
    shell_batch: int = 2048
    ce_in_main: bool = False
    seed: int = 0


def init_cert(key, cfg: CertConfig) -> CertParams:
    psi = init_mlp(key, [3, *cfg.hidden, cfg.feat_out], scale="lecun")
    return CertParams(
        psi=psi,
        alpha_raw=jp.asarray(_softplus_inv(cfg.alpha_init - cfg.alpha_floor)),
        eps_raw=jp.asarray(_softplus_inv(cfg.eps_init)),
    )


def _softplus_inv(y: float) -> float:
    y = max(y, 1e-6)
    return float(np.log(np.expm1(y)))


def alpha_of(p: CertParams, cfg: CertConfig):
    return cfg.alpha_floor + jax.nn.softplus(p.alpha_raw)


def epsp_of(p: CertParams):
    return jax.nn.softplus(p.eps_raw)


def ell_chart(z: jax.Array) -> jax.Array:
    """ell as a function of chart coords: ||E - I||_F = 2*sqrt(2)*sin(||z||/2)."""
    th = jp.linalg.norm(z, axis=-1)
    return 2.0 * jp.sqrt(2.0) * jp.sin(0.5 * th)


def V_chart(p: CertParams, cfg: CertConfig, z: jax.Array) -> jax.Array:
    a = alpha_of(p, cfg)
    th = jp.linalg.norm(z, axis=-1)
    quad = 8.0 * jp.sin(0.5 * th) ** 2
    if cfg.quad_only:
        return a**2 * quad
    feats = mlp_apply(p.psi, z, jax.nn.relu)
    feats0 = mlp_apply(p.psi, jp.zeros((1, 3)), jax.nn.relu)
    return jp.sum((feats - feats0) ** 2, axis=-1) + a**2 * quad


def W_chart(p: CertParams, cfg: CertConfig, z: jax.Array) -> jax.Array:
    return 1.0 - jp.exp(-epsp_of(p) * V_chart(p, cfg, z))


class LossPieces(NamedTuple):
    dec: jax.Array
    ce: jax.Array
    zubov: jax.Array
    bnd: jax.Array
    data: jax.Array
    reg: jax.Array


def loss_pieces(p: CertParams, cfg: CertConfig, z, zT, z_data, vm_data, z_ce=None, zT_ce=None) -> LossPieces:
    V0 = V_chart(p, cfg, z)
    V1 = V_chart(p, cfg, zT)
    epsp = epsp_of(p)
    W0 = 1.0 - jp.exp(-epsp * V0)
    W1 = 1.0 - jp.exp(-epsp * V1)
    lz = ell_chart(z)

    mask_dec = (lz >= cfg.eps_ball).astype(z.dtype)
    dec = jp.sum(jax.nn.relu(V1 - V0 + cfg.zeta_train) * mask_dec) / jp.clip(mask_dec.sum(), 1.0)

    zubov = jp.mean((W1 - 1.0 + (1.0 - W0) * jp.exp(epsp * lz)) ** 2)

    exits = (jp.linalg.norm(zT, axis=-1) > cfg.radius).astype(z.dtype)
    bnd = jp.sum(jax.nn.relu(1.0 - W0) ** 2 * exits) / jp.clip(exits.sum(), 1.0)

    W_target = 1.0 - jp.exp(-epsp * vm_data)
    data = jp.mean((W_chart(p, cfg, z_data) - W_target) ** 2)

    if z_ce is not None:
        Vc0 = V_chart(p, cfg, z_ce)
        Vc1 = V_chart(p, cfg, zT_ce)
        mc = (ell_chart(z_ce) >= cfg.eps_ball).astype(z.dtype)
        ce = jp.sum(jax.nn.relu(Vc1 - Vc0 + cfg.zeta_train) * mc) / jp.clip(mc.sum(), 1.0)
    else:
        ce = jp.zeros(())

    spec = sum(jax.nn.relu(jp.linalg.norm(w, ord=2) - cfg.spec_target) ** 2 for w, _ in p.psi)
    reg = cfg.eps_l2 * epsp**2 + cfg.w_spec * spec
    return LossPieces(dec, ce, zubov, bnd, data, reg)


def pgd_counterexamples(p: CertParams, cfg: CertConfig, T_chart, key, r_min=None):
    """Sign-gradient ascent on V(T z) - V(z) in the chart ball (P5 cadence)."""

    def gap(z):
        return V_chart(p, cfg, T_chart(z)) - V_chart(p, cfg, z)

    z = so3.sample_ball(key, cfg.pgd_batch, cfg.radius, r_min=r_min or 0.0)
    step = cfg.radius / cfg.pgd_steps

    def body(z, _):
        g = jax.grad(lambda zz: gap(zz).sum())(z)
        z = z + step * jp.sign(g)
        nrm = jp.linalg.norm(z, axis=-1, keepdims=True)
        z = z * jp.minimum(1.0, cfg.radius / jp.clip(nrm, 1e-12))
        return z, None

    z, _ = jax.lax.scan(body, z, None, length=cfg.pgd_steps)
    return z, gap(z)


def train_certificate(T_chart, vm_fn, cfg: CertConfig, verbose=True):
    """Phase 2a loop (code plan Sec. 5.2/5.3). T_chart: (n,3)->(n,3) frozen"""
    if cfg.ce_batch > cfg.batch // 4:
        cfg = cfg._replace(ce_batch=max(1, cfg.batch // 4))
    key = jax.random.key(cfg.seed)
    key, k0, kd = jax.random.split(key, 3)
    p = init_cert(k0, cfg)

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(p)

    key, kb = jax.random.split(key)
    buf = so3.sample_ball(kb, cfg.buffer_size, cfg.radius)
    z_data = so3.sample_ball(kd, 4096, cfg.radius)
    vm_data = vm_fn(z_data)

    ce = jp.zeros((0, 3))
    rho_cur = np.inf

    lam = {"dec": 1.0, "bnd": 1.0}
    w_z_sched = lambda i: min(1.0, i / max(1, int(0.2 * cfg.iters)))

    @jax.jit
    def step_fn(p, opt_state, zb, zb_ce, z_shell, z_data_b, vm_b, lam_dec, lam_bnd, w_z):
        zT = T_chart(zb)
        zT_ce = T_chart(zb_ce)

        def total(p):
            L = loss_pieces(p, cfg, zb, zT, z_data_b, vm_b, zb_ce, zT_ce)
            lam_data = jp.exp(-0.5 * (lam_dec + lam_bnd) / 10.0) * cfg.w_data
            V_sh = V_chart(p, cfg, z_shell)
            med = jax.lax.stop_gradient(jp.median(V_sh)) + 1e-9
            shell = jp.mean(jax.nn.relu(1.0 - V_sh / med) ** 2)
            return (
                lam_dec * L.dec
                + cfg.w_ce * L.ce
                + w_z * L.zubov
                + lam_bnd * L.bnd
                + lam_data * L.data
                + cfg.w_shell * shell
                + L.reg,
                (L, shell),
            )

        (val, (L, shell)), g = jax.value_and_grad(total, has_aux=True)(p)
        upd, opt_state = opt.update(g, opt_state)
        p = jax.tree.map(lambda a, b: a + b, p, upd)
        return p, opt_state, L, shell

    hist = []
    for it in range(cfg.iters):
        key, k1, k2, k3 = jax.random.split(key, 4)
        if it % cfg.resample_every == 0 and it > 0:
            key, kr = jax.random.split(key)
            n_new = cfg.buffer_size // 4
            buf = jp.concatenate([buf[n_new:], so3.sample_ball(kr, n_new, cfg.radius)])
        if cfg.mine_scoped and it % cfg.rho_refresh_every == 0:
            key, ks = jax.random.split(key)
            z_sh = so3.sample_ball(ks, 16384, cfg.radius + cfg.alpha_max, r_min=cfg.radius)
            rho_cur = float(jp.min(V_chart(p, cfg, z_sh)))

        if cfg.mine_keep > 0 and it % cfg.pgd_every == 0:
            z_ce, gaps = pgd_counterexamples(p, cfg, T_chart, k3)
            hits = z_ce[gaps > -cfg.zeta_train]
            key, km = jax.random.split(key)
            z_rnd = so3.sample_ball(km, cfg.mine_batch, cfg.radius)
            gap_rnd = V_chart(p, cfg, T_chart(z_rnd)) - V_chart(p, cfg, z_rnd)
            if cfg.mine_scoped:
                in_level = V_chart(p, cfg, z_rnd) <= rho_cur
                gap_rnd = jp.where(in_level, gap_rnd, -jp.inf)
                hits_V = V_chart(p, cfg, hits) if hits.shape[0] else None
                if hits_V is not None:
                    hits = hits[hits_V <= rho_cur]
            worst = jp.argsort(gap_rnd)[-cfg.mine_keep:]
            mined = z_rnd[worst][gap_rnd[worst] > -cfg.zeta_train]
            hits = jp.concatenate([hits, mined])
            if hits.shape[0] > 0:
                ce = jp.concatenate([ce, hits])[-cfg.ce_buffer_max:]
        elif cfg.mine_keep == 0 and it % cfg.pgd_every == 0:
            z_ce, gaps = pgd_counterexamples(p, cfg, T_chart, k3)
            hits = z_ce[gaps > -cfg.zeta_train]
            if hits.shape[0] > 0:
                ce = jp.concatenate([ce, hits])[-8192:]

        idx = jax.random.randint(k1, (cfg.batch - cfg.ce_batch,), 0, cfg.buffer_size)
        zb = buf[idx]
        if ce.shape[0] > 0:
            cidx = jax.random.randint(k2, (cfg.ce_batch,), 0, ce.shape[0])
            zb_ce = ce[cidx]
        else:
            zb_ce = zb[: cfg.ce_batch]
        if cfg.ce_in_main:
            zb = jp.concatenate([zb, zb_ce])
        di = jax.random.randint(k2, (1024,), 0, z_data.shape[0])
        key, ksh = jax.random.split(key)
        z_shell = so3.sample_ball(ksh, cfg.shell_batch, cfg.radius + cfg.alpha_max, r_min=cfg.radius)

        p, opt_state, L, shell_loss = step_fn(
            p, opt_state, zb, zb_ce, z_shell, z_data[di], vm_data[di],
            lam["dec"], lam["bnd"], w_z_sched(it),
        )

        if cfg.primal_dual and it % 20 == 0:
            lam["dec"] = float(np.clip(lam["dec"] + cfg.lam_lr * float(L.dec) * 20, 0, cfg.lam_max))
            lam["bnd"] = float(np.clip(lam["bnd"] + cfg.lam_lr * float(L.bnd) * 20, 0, cfg.lam_max))

        if it % 500 == 0 or it == cfg.iters - 1:
            rec = dict(
                it=it,
                dec=float(L.dec),
                ce_loss=float(L.ce),
                shell=float(shell_loss),
                rho_cur=float(rho_cur) if np.isfinite(rho_cur) else None,
                zubov=float(L.zubov),
                bnd=float(L.bnd),
                data=float(L.data),
                lam_dec=lam["dec"],
                lam_bnd=lam["bnd"],
                epsp=float(epsp_of(p)),
                alpha=float(alpha_of(p, cfg)),
                n_ce=int(ce.shape[0]),
            )
            hist.append(rec)
            if verbose:
                print(
                    f"  it {it:5d} dec {rec['dec']:.2e} ce {rec['ce_loss']:.2e} "
                    f"zub {rec['zubov']:.2e} bnd {rec['bnd']:.2e} data {rec['data']:.2e} "
                    f"lam ({rec['lam_dec']:.1f},{rec['lam_bnd']:.1f}) "
                    f"eps' {rec['epsp']:.3f} alpha {rec['alpha']:.3f} CE {rec['n_ce']}"
                )
        if any(not np.isfinite(v) for v in (float(L.dec), float(L.zubov), float(L.bnd))):
            raise FloatingPointError(f"NaN/inf in losses at iter {it} — sampler bug by construction")

    return p, hist, ce
