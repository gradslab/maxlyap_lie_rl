"""Copy of certificate.train_certificate with switches for the ablations (results/rx). With the defaults
(w_rec=1, data_fixed=None, n_data=4096, eps_auto=False) it runs the paper's recipe unchanged; the paper's
module is not modified. Generated from lie_rl/certificate.py (line 312 onwards) by exact text substitution.
"""
import jax
import jax.numpy as jp
import numpy as np
import optax

from .certificate import (CertConfig, V_chart, cost_of, _softplus_inv, init_cert, loss_pieces, make_ell, make_vm,
                          pgd_counterexamples, sample_ball, sample_mixed)


def train_certificate(T_chart, G, cfg: CertConfig, verbose=True, w_rec=1.0, data_fixed=None, n_data=4096, eps_auto=False, rec_rel=False,
                      p_init=None, ce_pinned=None, exit_pinned=None, ramp=True):
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
    z_data = sample_ball(kd, n_data, cfg.m, cfg.radius)
    vm_data = vm_fn(z_data)
    if eps_auto:
        cfg = cfg._replace(eps_fixed=float(1.0 / np.median(np.asarray(vm_data))))
        p = p._replace(eps_raw=jp.asarray(_softplus_inv(cfg.eps_fixed)))
        opt_state = opt.init(p)
    if p_init is not None:
        p = p_init
        opt_state = opt.init(p)
    ce = jp.zeros((0, cfg.m))
    ce_exit = jp.zeros((0, cfg.m))
    n_pin = cfg.ce_batch // 2
    lam = {"dec": 1.0, "bnd": 1.0 if cfg.use_bnd else 0.0}
    rho_cur = jp.inf

    @jax.jit
    def step_fn(p, opt_state, zb, zb_ce, zb_exit, z_data_b, vm_b, z_shell, lam_dec, lam_bnd, w_z, w_exit):
        zT = T_chart(zb)
        zT_ce = T_chart(zb_ce)

        def total(pp):
            L = loss_pieces(pp, cfg, ell, zb, zT, z_data_b, vm_b, zb_ce, zT_ce, zb_exit)
            lam_data = (jp.exp(-0.5 * (lam_dec + lam_bnd) / 10.0) * cfg.w_data) if data_fixed is None else data_fixed
            V_sh = V_chart(pp, cfg, ell, z_shell)
            med = jax.lax.stop_gradient(jp.median(V_sh)) + 1e-9
            shell = jp.mean(jax.nn.relu(1.0 - V_sh / med) ** 2)
            if rec_rel:
                Vr0 = V_chart(pp, cfg, ell, zb); Vr1 = V_chart(pp, cfg, ell, zT); cr = cost_of(cfg, ell, zb)
                rel = jp.mean(((Vr1 - Vr0 + cr) / (jax.lax.stop_gradient(Vr0) + cr + 1e-6)) ** 2)
            else:
                rel = 0.0
            return (lam_dec * L.dec + cfg.w_ce * L.ce + w_z * L.zubov + (w_z * rel if rec_rel else 0.0)
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
        if ce_pinned is not None and ce_pinned.shape[0] > 0:
            key, kp = jax.random.split(key)
            zb_ce = jp.concatenate([ce_pinned[jax.random.randint(kp, (n_pin,), 0, ce_pinned.shape[0])], zb_ce[n_pin:]])
        if ce_exit.shape[0] > 0:
            edx = jax.random.randint(k3, (cfg.ce_batch,), 0, ce_exit.shape[0])
            zb_exit = ce_exit[edx]; w_exit_eff = cfg.w_exit
        else:
            zb_exit = zb[:cfg.ce_batch]; w_exit_eff = 0.0
        if exit_pinned is not None and exit_pinned.shape[0] > 0:
            key, kq = jax.random.split(key)
            zb_exit = jp.concatenate([exit_pinned[jax.random.randint(kq, (n_pin,), 0, exit_pinned.shape[0])], zb_exit[n_pin:]])
            w_exit_eff = cfg.w_exit
        ddx = jax.random.randint(k3, (1024,), 0, z_data.shape[0])
        w_z = w_rec * (min(1.0, it / max(1, int(0.2 * cfg.iters))) if ramp else 1.0)

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
    return p, hist, cfg
