"""JAX PPO for the generalized Lie group task."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax

from . import env as E
from .nets import init_mlp, mlp_apply


class PPOConfig(NamedTuple):
    n_envs: int = 32
    n_steps: int = 100
    n_minibatches: int = 8
    n_epochs: int = 20
    n_total_steps: int = 400_000
    actor_lr: float = 9.16e-4
    critic_lr: float = 2.18e-3
    clip_coef: float = 0.130
    ent_coef: float = 0.013
    vf_coef: float = 0.740
    max_grad_norm: float = 0.602
    gamma: float = 0.916
    gae_lambda: float = 0.800
    norm_adv: bool = True
    clip_vloss: bool = True
    logstd_init: float = 0.0
    hidden: int = 64


class AgentParams(NamedTuple):
    actor: list
    logstd: jax.Array
    critic: list


def actor_mean(actor_params, obs):
    return mlp_apply(actor_params, obs, jp.tanh, out_act=jp.tanh)


def critic_value(critic_params, obs):
    return mlp_apply(critic_params, obs, jp.tanh)[..., 0]


def init_agent(key, cfg: PPOConfig, obs_dim: int, act_dim: int) -> AgentParams:
    ka, kc = jax.random.split(key)
    h = cfg.hidden
    return AgentParams(
        actor=init_mlp(ka, [obs_dim, h, h, act_dim], out_gain=0.01),
        logstd=jp.full((act_dim,), cfg.logstd_init),
        critic=init_mlp(kc, [obs_dim, h, h, 1], out_gain=1.0),
    )


def _logp_gauss(mean, logstd, a):
    var = jp.exp(2 * logstd)
    return (-0.5 * ((a - mean) ** 2 / var + 2 * logstd + jp.log(2 * jp.pi))).sum(-1)


def train_ppo(seed: int, cfg: PPOConfig, env_params: E.EnvParams, verbose: bool = True):
    G = env_params.G
    key = jax.random.key(seed)
    key, k_init, k_env = jax.random.split(key, 3)
    agent = init_agent(k_init, cfg, G.obs_dim, G.m)

    opt_a = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.actor_lr))
    opt_c = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.critic_lr))

    def opt_init(agent):
        return (opt_a.init((agent.actor, agent.logstd)), opt_c.init(agent.critic))

    def opt_update(g: AgentParams, opt_state, agent: AgentParams):
        sa, sc = opt_state
        ua, sa = opt_a.update((g.actor, g.logstd), sa)
        uc, sc = opt_c.update(g.critic, sc)
        return AgentParams(actor=ua[0], logstd=ua[1], critic=uc), (sa, sc)

    opt_state = opt_init(agent)
    state = E.reset(k_env, cfg.n_envs, env_params)

    batch = cfg.n_envs * cfg.n_steps
    n_iters = cfg.n_total_steps // batch
    mb_size = batch // cfg.n_minibatches

    @jax.jit
    def rollout(agent, state, key):
        def one_step(carry, _):
            state, key = carry
            key, k_act = jax.random.split(key)
            obs = E.obs(G, state.E)
            mean = actor_mean(agent.actor, obs)
            a = mean + jp.exp(agent.logstd) * jax.random.normal(k_act, mean.shape)
            logp = _logp_gauss(mean, agent.logstd, a)
            value = critic_value(agent.critic, obs)
            new_state, E_next, reward, trunc = E.step(state, a, env_params)
            v_next = critic_value(agent.critic, E.obs(G, E_next))
            return (new_state, key), (obs, a, logp, value, reward, trunc, v_next)

        (state, key), traj = jax.lax.scan(one_step, (state, key), None, length=cfg.n_steps)
        return state, key, traj

    @jax.jit
    def gae(traj):
        obs, a, logp, values, rewards, truncs, v_next = traj

        def back(adv_next, t):
            delta = rewards[t] + cfg.gamma * v_next[t] - values[t]
            adv = delta + cfg.gamma * cfg.gae_lambda * (1.0 - truncs[t]) * adv_next
            return adv, adv

        T = rewards.shape[0]
        _, advs = jax.lax.scan(back, jp.zeros_like(rewards[0]), jp.arange(T - 1, -1, -1))
        advs = advs[::-1]
        return advs, advs + values

    @jax.jit
    def update(agent, opt_state, flat, key):
        obs, act, logp_old, val_old, advs, rets = flat

        def loss_fn(agent, idx):
            o, a, lp0, v0, adv, ret = (x[idx] for x in (obs, act, logp_old, val_old, advs, rets))
            mean = actor_mean(agent.actor, o)
            lp = _logp_gauss(mean, agent.logstd, a)
            ratio = jp.exp(lp - lp0)
            if cfg.norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = jp.maximum(pg1, pg2).mean()
            v = critic_value(agent.critic, o)
            if cfg.clip_vloss:
                v_clip = v0 + jp.clip(v - v0, -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * jp.maximum((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            else:
                v_loss = 0.5 * ((v - ret) ** 2).mean()
            ent = (agent.logstd + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
            return pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

        def epoch(carry, _):
            agent, opt_state, key = carry
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, batch)

            def mb(carry, i):
                agent, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb_size, mb_size)
                g = jax.grad(loss_fn)(agent, idx)
                upd, opt_state = opt_update(g, opt_state, agent)
                agent = jax.tree.map(lambda p, u: p + u, agent, upd)
                return (agent, opt_state), None

            (agent, opt_state), _ = jax.lax.scan(mb, (agent, opt_state), jp.arange(cfg.n_minibatches))
            return (agent, opt_state, key), None

        (agent, opt_state, key), _ = jax.lax.scan(epoch, (agent, opt_state, key), None, length=cfg.n_epochs)
        return agent, opt_state, key

    hist = []
    for it in range(n_iters):
        state, key, traj = rollout(agent, state, key)
        advs, rets = gae(traj)
        obs, a, logp, values, rewards, truncs, v_next = traj
        flat = tuple(x.reshape((batch,) + x.shape[2:]) for x in (obs, a, logp, values, advs, rets))
        agent, opt_state, key = update(agent, opt_state, flat, key)
        hist.append(float(rewards.sum() / truncs.sum().clip(1)))
        if verbose and (it % 10 == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}/{n_iters}  mean episode return ~ {hist[-1]:8.3f}", flush=True)
    return agent, hist


def eval_policy(agent: AgentParams, key, n_episodes: int, env_params: E.EnvParams):
    """Deterministic mean-policy evaluation over the working region."""
    G = env_params.G
    state = E.reset(key, n_episodes, env_params)
    pol = lambda Ex: actor_mean(agent.actor, E.obs(G, Ex))
    Ef, ret = E.rollout(G, pol, state.E, env_params)
    th, tn = G.err_parts(Ef)
    return np.asarray(ret), np.asarray(th), np.asarray(tn)
