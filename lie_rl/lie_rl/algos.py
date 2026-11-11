"""TRPO, SAC and TD3 on the general Lie group task."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax

from . import env as E
from .nets import init_mlp, mlp_apply
from .ppo import AgentParams, actor_mean, _logp_gauss


class TRPOConfig(NamedTuple):
    n_envs: int = 32
    n_steps: int = 100
    n_total_steps: int = 400_000
    gamma: float = 0.916
    gae_lambda: float = 0.800
    delta: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    bt_ratio: float = 0.5
    bt_max: int = 10
    critic_lr: float = 2.18e-3
    critic_epochs: int = 20
    logstd_init: float = 0.0
    hidden: int = 64


def train_trpo(seed, cfg: TRPOConfig, env_params: E.EnvParams, verbose=False):
    G = env_params.G
    key = jax.random.key(seed)
    key, ki, ke = jax.random.split(key, 3)
    ka, kc = jax.random.split(ki)
    h = cfg.hidden
    agent = AgentParams(
        actor=init_mlp(ka, [G.obs_dim, h, h, G.m], out_gain=0.01),
        logstd=jp.full((G.m,), cfg.logstd_init),
        critic=init_mlp(kc, [G.obs_dim, h, h, 1], out_gain=1.0),
    )
    copt = optax.adam(cfg.critic_lr)
    cstate = copt.init(agent.critic)
    state = E.reset(ke, cfg.n_envs, env_params)
    batch = cfg.n_envs * cfg.n_steps
    n_iters = cfg.n_total_steps // batch

    def critic_value(cp, o):
        return mlp_apply(cp, o, jp.tanh)[..., 0]

    @jax.jit
    def rollout(agent, state, key):
        def one(carry, _):
            state, key = carry
            key, kact = jax.random.split(key)
            o = E.obs(G, state.E)
            mu = actor_mean(agent.actor, o)
            a = mu + jp.exp(agent.logstd) * jax.random.normal(kact, mu.shape)
            v = critic_value(agent.critic, o)
            ns, En, r, tr = E.step(state, a, env_params)
            vn = critic_value(agent.critic, E.obs(G, En))
            return (ns, key), (o, a, v, r, tr, vn, mu)
        (state, key), tr = jax.lax.scan(one, (state, key), None, length=cfg.n_steps)
        return state, key, tr

    @jax.jit
    def gae(tr):
        o, a, v, r, trunc, vn, mu = tr
        def back(adv_next, t):
            delta = r[t] + cfg.gamma * vn[t] - v[t]
            adv = delta + cfg.gamma * cfg.gae_lambda * (1.0 - trunc[t]) * adv_next
            return adv, adv
        T = r.shape[0]
        _, advs = jax.lax.scan(back, jp.zeros_like(r[0]), jp.arange(T - 1, -1, -1))
        advs = advs[::-1]
        return advs, advs + v

    @jax.jit
    def trpo_step(agent, o, a, adv, logp_old, mu_old):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def surrogate(actor):
            mu = actor_mean(actor, o)
            lp = _logp_gauss(mu, agent.logstd, a)
            return jp.mean(jp.exp(lp - logp_old) * adv)

        def mean_kl(actor):
            mu = actor_mean(actor, o)
            return 0.5 * jp.mean(jp.sum(((mu - mu_old) / jp.exp(agent.logstd)) ** 2, -1))

        g = jax.grad(surrogate)(agent.actor)
        flat, unflat = jax.flatten_util.ravel_pytree(g)

        def fvp(x):
            xt = unflat(x)
            def kl_dot(actor):
                return sum(jp.vdot(gi, xi) for gi, xi in
                           zip(jax.tree.leaves(jax.grad(mean_kl)(actor)), jax.tree.leaves(xt)))
            hv, _ = jax.flatten_util.ravel_pytree(jax.grad(kl_dot)(agent.actor))
            return hv + cfg.cg_damping * x

        x = jp.zeros_like(flat)
        r_ = flat
        p = flat
        rr = jp.vdot(r_, r_)

        def cg_body(i, st):
            x, r_, p, rr = st
            Ap = fvp(p)
            alpha = rr / (jp.vdot(p, Ap) + 1e-12)
            x = x + alpha * p
            r_ = r_ - alpha * Ap
            rr_new = jp.vdot(r_, r_)
            p = r_ + (rr_new / (rr + 1e-12)) * p
            return (x, r_, p, rr_new)

        x, _, _, _ = jax.lax.fori_loop(0, cfg.cg_iters, cg_body, (x, r_, p, rr))
        shs = 0.5 * jp.vdot(x, fvp(x))
        step = jp.sqrt(cfg.delta / (shs + 1e-12)) * x
        s0 = surrogate(agent.actor)

        def bt_body(carry):
            i, frac, done, best = carry
            cand = jax.tree.map(lambda p_, d_: p_ + frac * d_, agent.actor, unflat(step))
            ok = (surrogate(cand) > s0) & (mean_kl(cand) < cfg.delta * 1.5)
            best = jax.tree.map(lambda b, c: jp.where(ok & ~done, c, b), best, cand)
            return (i + 1, frac * cfg.bt_ratio, done | ok, best)

        _, _, _, new_actor = jax.lax.while_loop(
            lambda c: (c[0] < cfg.bt_max) & (~c[2]),
            bt_body, (0, 1.0, False, agent.actor))
        return agent._replace(actor=new_actor)

    @jax.jit
    def critic_fit(critic, cstate, o, ret):
        def loss(cp):
            return jp.mean((critic_value(cp, o) - ret) ** 2)
        def one(carry, _):
            cp, cs = carry
            g = jax.grad(loss)(cp)
            u, cs = copt.update(g, cs, cp)
            return (optax.apply_updates(cp, u), cs), None
        (critic, cstate), _ = jax.lax.scan(one, (critic, cstate), None, length=cfg.critic_epochs)
        return critic, cstate

    hist = []
    for it in range(n_iters):
        state, key, tr = rollout(agent, state, key)
        advs, rets = gae(tr)
        o, a, v, r, trunc, vn, mu = tr
        f = lambda z: z.reshape((batch,) + z.shape[2:])
        o, a, advs, rets, mu = map(f, (o, a, advs, rets, mu))
        logp_old = _logp_gauss(mu, agent.logstd, a)
        agent = trpo_step(agent, o, a, advs, logp_old, mu)
        critic, cstate = critic_fit(agent.critic, cstate, o, rets)
        agent = agent._replace(critic=critic)
        hist.append(float(r.sum() / trunc.sum().clip(1)))
        if verbose and it % 10 == 0:
            print(f"  iter {it}/{n_iters} return ~ {hist[-1]:.3f}", flush=True)
    return agent, hist


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.o = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, act_dim), np.float32)
        self.r = np.zeros((capacity,), np.float32)
        self.o2 = np.zeros((capacity, obs_dim), np.float32)
        self.cap = capacity
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2):
        n = o.shape[0]
        idx = (self.ptr + np.arange(n)) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx] = o, a, r, o2
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = int(min(self.size + n, self.cap))

    def sample(self, rng, batch):
        i = rng.integers(0, self.size, size=batch)
        return (jp.asarray(self.o[i]), jp.asarray(self.a[i]),
                jp.asarray(self.r[i]), jp.asarray(self.o2[i]))


def _critic_apply(c, o, a):
    return mlp_apply(c, jp.concatenate([o, a], -1), jp.tanh)[..., 0]


def _polyak(t, s, tau):
    return jax.tree.map(lambda a, b: (1 - tau) * a + tau * b, t, s)


class SACConfig(NamedTuple):
    n_steps: int = 128_000
    n_envs: int = 32
    actor_lr: float = 2.24e-3
    critic_lr: float = 9.42e-4
    gamma: float = 0.975
    tau: float = 8.76e-3
    batch: int = 8192
    buffer: int = 128_000
    target_period: int = 40
    train_period: int = 100
    train_steps: int = 20
    grad_clip: float = 2.5
    eps: float = 6.34e-7
    alpha_init: float = 0.045
    alpha_lr: float = 2.86e-2
    hidden: int = 64
    critic_hidden: int = 256
    logstd_min: float = -5.0
    logstd_max: float = 2.0


def train_sac(seed, cfg: SACConfig, env_params: E.EnvParams, verbose=False):
    G = env_params.G
    D, NA = G.obs_dim, G.m
    target_entropy = -float(NA)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    key, km, kl, k1, k2, ke = jax.random.split(key, 6)
    h, ch = cfg.hidden, cfg.critic_hidden
    mean_net = init_mlp(km, [D, h, h, NA], out_gain=0.01)
    logstd_net = init_mlp(kl, [D, h, h, NA], out_gain=0.01)
    c1 = init_mlp(k1, [D + NA, ch, ch, 1], out_gain=1.0)
    c2 = init_mlp(k2, [D + NA, ch, ch, 1], out_gain=1.0)
    c1t, c2t = c1, c2

    aopt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                       optax.adamw(cfg.actor_lr, eps=cfg.eps, weight_decay=0.0))
    copt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                       optax.adamw(cfg.critic_lr, eps=cfg.eps, weight_decay=0.0))
    lopt = optax.adam(cfg.alpha_lr)
    as_ = aopt.init((mean_net, logstd_net))
    cs = copt.init((c1, c2))
    log_alpha = jp.asarray(np.log(cfg.alpha_init, dtype=np.float32))
    ls = lopt.init(log_alpha)

    def mean_of(mn, o):
        return mlp_apply(mn, o, jp.tanh)

    def sample_and_logp(mn, ln, o, k):
        mu = mean_of(mn, o)
        log_std = jp.clip(mlp_apply(ln, o, jp.tanh), cfg.logstd_min, cfg.logstd_max)
        std = jp.exp(log_std)
        u = mu + std * jax.random.normal(k, mu.shape)
        a = jp.tanh(u)
        lg = -0.5 * (((u - mu) / std) ** 2 + 2.0 * log_std + jp.log(2.0 * jp.pi))
        return a, jp.sum(lg - jp.log(1.0 - a ** 2 + 1e-6), axis=-1)

    @jax.jit
    def critic_step(mn, ln, c1, c2, cs, c1t, c2t, log_alpha, o, a, r, o2, k):
        a2, lp2 = sample_and_logp(mn, ln, o2, k)
        nq = jp.minimum(_critic_apply(c1t, o2, a2), _critic_apply(c2t, o2, a2)) \
             - jp.exp(log_alpha) * lp2
        y = r + cfg.gamma * nq
        def loss(cc):
            return (jp.mean((y - _critic_apply(cc[0], o, a)) ** 2)
                    + jp.mean((y - _critic_apply(cc[1], o, a)) ** 2))
        g = jax.grad(loss)((c1, c2))
        u, cs = copt.update(g, cs, (c1, c2))
        c1, c2 = optax.apply_updates((c1, c2), u)
        return c1, c2, cs

    @jax.jit
    def actor_step(mn, ln, as_, c1, c2, log_alpha, o, k):
        def loss(p):
            a, lp = sample_and_logp(p[0], p[1], o, k)
            q = jp.minimum(_critic_apply(c1, o, a), _critic_apply(c2, o, a))
            return jp.mean(jp.exp(log_alpha) * lp - q), lp
        (_, lp), g = jax.value_and_grad(loss, has_aux=True)((mn, ln))
        u, as_ = aopt.update(g, as_, (mn, ln))
        mn, ln = optax.apply_updates((mn, ln), u)
        return mn, ln, as_, lp

    @jax.jit
    def alpha_step(log_alpha, ls, lp):
        def loss(la):
            return -jp.mean(la * (jax.lax.stop_gradient(lp) + target_entropy))
        g = jax.grad(loss)(log_alpha)
        u, ls = lopt.update(g, ls, log_alpha)
        return optax.apply_updates(log_alpha, u), ls

    buf = ReplayBuffer(cfg.buffer, D, NA)
    state = E.reset(ke, cfg.n_envs, env_params)
    obs = np.asarray(E.obs(G, state.E))
    n_samples = last_train = 0
    hist = []
    while n_samples < cfg.n_steps:
        key, ks = jax.random.split(key)
        act = np.asarray(sample_and_logp(mean_net, logstd_net, jp.asarray(obs), ks)[0])
        state, E_next, rew, _ = E.step(state, jp.asarray(act), env_params)
        obs2 = np.asarray(E.obs(G, E_next))
        buf.add(obs, act, np.asarray(rew), obs2)
        obs = np.asarray(E.obs(G, state.E))
        n_samples += cfg.n_envs
        hist.append(float(np.mean(rew)) * env_params.max_steps)
        if buf.size >= cfg.batch and n_samples - last_train >= cfg.train_period:
            last_train = n_samples
            for j in range(cfg.train_steps):
                key, kc, ka2 = jax.random.split(key, 3)
                bo, ba, br, bo2 = buf.sample(rng, cfg.batch)
                c1, c2, cs = critic_step(mean_net, logstd_net, c1, c2, cs,
                                         c1t, c2t, log_alpha, bo, ba, br, bo2, kc)
                mean_net, logstd_net, as_, lp = actor_step(
                    mean_net, logstd_net, as_, c1, c2, log_alpha, bo, ka2)
                log_alpha, ls = alpha_step(log_alpha, ls, lp)
                if j % max(1, cfg.target_period // cfg.train_steps) == 0:
                    c1t = _polyak(c1t, c1, cfg.tau)
                    c2t = _polyak(c2t, c2, cfg.tau)
        if verbose and n_samples % 12800 == 0:
            print(f"  {n_samples}/{cfg.n_steps}", flush=True)
    return AgentParams(actor=mean_net, logstd=jp.zeros((NA,)), critic=c1), hist


class TD3Config(NamedTuple):
    n_steps: int = 128_000
    n_envs: int = 32
    actor_lr: float = 8.32e-3
    critic_lr: float = 5.92e-3
    gamma: float = 0.823
    tau: float = 0.0211
    batch: int = 2048
    buffer: int = 128_000
    actor_target_period: int = 40
    critic_target_period: int = 20
    train_period: int = 200
    train_steps: int = 20
    grad_clip: float = 5.0
    target_smooth: float = 0.1
    weight_decay: float = 0.01
    expl_noise: float = 0.1
    hidden: int = 64
    critic_hidden: int = 256


def train_td3(seed, cfg: TD3Config, env_params: E.EnvParams, verbose=False):
    G = env_params.G
    D, NA = G.obs_dim, G.m
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    key, ka, k1, k2, ke = jax.random.split(key, 5)
    h, ch = cfg.hidden, cfg.critic_hidden
    actor = init_mlp(ka, [D, h, h, NA], out_gain=0.01)
    c1 = init_mlp(k1, [D + NA, ch, ch, 1], out_gain=1.0)
    c2 = init_mlp(k2, [D + NA, ch, ch, 1], out_gain=1.0)
    actor_t, c1t, c2t = actor, c1, c2

    aopt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                       optax.adamw(cfg.actor_lr, weight_decay=cfg.weight_decay))
    copt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                       optax.adamw(cfg.critic_lr, weight_decay=cfg.weight_decay))
    as_ = aopt.init(actor)
    cs = copt.init((c1, c2))

    @jax.jit
    def critic_step(actor_t, c1, c2, cs, c1t, c2t, o, a, r, o2, k):
        a2 = actor_mean(actor_t, o2)
        noise = jp.clip(cfg.target_smooth * jax.random.normal(k, a2.shape), -0.5, 0.5)
        a2 = jp.clip(a2 + noise, -1.0, 1.0)
        y = r + cfg.gamma * jp.minimum(_critic_apply(c1t, o2, a2),
                                       _critic_apply(c2t, o2, a2))
        def loss(cc):
            return (jp.mean((y - _critic_apply(cc[0], o, a)) ** 2)
                    + jp.mean((y - _critic_apply(cc[1], o, a)) ** 2))
        g = jax.grad(loss)((c1, c2))
        u, cs = copt.update(g, cs, (c1, c2))
        c1, c2 = optax.apply_updates((c1, c2), u)
        return c1, c2, cs

    @jax.jit
    def actor_step(actor, as_, c1, o):
        def loss(ap):
            return -jp.mean(_critic_apply(c1, o, actor_mean(ap, o)))
        g = jax.grad(loss)(actor)
        u, as_ = aopt.update(g, as_, actor)
        return optax.apply_updates(actor, u), as_

    buf = ReplayBuffer(cfg.buffer, D, NA)
    state = E.reset(ke, cfg.n_envs, env_params)
    obs = np.asarray(E.obs(G, state.E))
    n_samples = last_train = 0
    hist = []
    while n_samples < cfg.n_steps:
        key, kn = jax.random.split(key)
        a = actor_mean(actor, jp.asarray(obs))
        a = np.asarray(jp.clip(a + cfg.expl_noise * jax.random.normal(kn, a.shape), -1, 1))
        state, E_next, rew, _ = E.step(state, jp.asarray(a), env_params)
        obs2 = np.asarray(E.obs(G, E_next))
        buf.add(obs, a, np.asarray(rew), obs2)
        obs = np.asarray(E.obs(G, state.E))
        n_samples += cfg.n_envs
        hist.append(float(np.mean(rew)) * env_params.max_steps)
        if buf.size >= cfg.batch and n_samples - last_train >= cfg.train_period:
            last_train = n_samples
            for j in range(cfg.train_steps):
                key, kc = jax.random.split(key)
                bo, ba, br, bo2 = buf.sample(rng, cfg.batch)
                c1, c2, cs = critic_step(actor_t, c1, c2, cs, c1t, c2t, bo, ba, br, bo2, kc)
                if j % 2 == 0:
                    actor, as_ = actor_step(actor, as_, c1, bo)
                if j % max(1, cfg.critic_target_period // cfg.train_steps) == 0:
                    c1t = _polyak(c1t, c1, cfg.tau)
                    c2t = _polyak(c2t, c2, cfg.tau)
                if j % max(1, cfg.actor_target_period // cfg.train_steps) == 0:
                    actor_t = _polyak(actor_t, actor, cfg.tau)
        if verbose and n_samples % 12800 == 0:
            print(f"  {n_samples}/{cfg.n_steps}", flush=True)
    return AgentParams(actor=actor, logstd=jp.zeros((NA,)), critic=c1), hist


ALGOS = {"trpo": (train_trpo, TRPOConfig), "sac": (train_sac, SACConfig),
         "td3": (train_td3, TD3Config)}
