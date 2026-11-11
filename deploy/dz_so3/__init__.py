"""DZ-SO3: discrete-time Zubov certification of tangent-space RL policies on SO(3)."""

import jax

jax.config.update("jax_enable_x64", True)

from . import certificate, env, ground_truth, nets, ppo, so3, tier0, tier1a, tier2
