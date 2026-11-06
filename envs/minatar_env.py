"""MinAtar (Young & Tian 2019) via gymnax: Asterix, Breakout, Freeway, Seaquest, SpaceInvaders.

Observation: the 10x10 grid of binary channels (4 / 4 / 7 / 10 / 6 channels), cell-major flat vector of size
10*10*C (plane-tokenized like Sokoban, readout action head). No goal vector (goal_size 0).
Actions: the game's minimal action set (Asterix 5, Breakout 3, Freeway 3, Seaquest 6, SpaceInvaders 4).
Reward: the game's own. done = game over; the time limit is the EpisodeWrapper's (gymnax's own limit is
turned off so a time-out is a truncation, not a terminal). `success` is always 0 (no solve notion).
eval_reset = reset (no held-out split).
"""

import gymnax
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from envs.utils import State

GAMES = ('asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders')
GYMNAX_IDS = dict(zip(GAMES, ('Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar', 'Seaquest-MinAtar',
                                'SpaceInvaders-MinAtar')))


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(game='breakout', episode_length=1000)


class MinAtarEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self._env, params = gymnax.make(GYMNAX_IDS[config.game])
        self._params = params.replace(max_steps_in_episode=2**30)  # EpisodeWrapper owns the time limit
        self.m, self.n, self.obs_channels = self._env.observation_space(self._params).shape
        self.action_head = 'readout'

    @property
    def action_size(self): return self._env.num_actions

    @property
    def observation_size(self): return self.m * self.n * self.obs_channels

    @property
    def goal_size(self): return 0

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        obs, data = self._env.reset_env(key, self._params)
        return State(
            data=data, obs=obs.reshape(-1).astype(jnp.float32),
            reward=jnp.array(0.0, jnp.float32), done=jnp.array(0.0, jnp.float32),
            metrics={'success': jnp.array(0.0, jnp.float32), 'reward': jnp.array(0.0, jnp.float32)},
            info={'rng': rng, 'target_goal': jnp.zeros((0,), jnp.float32)},
        )

    eval_reset = reset

    def step(self, state: State, action: jax.Array) -> State:
        rng, key = jax.random.split(state.info['rng'])
        obs, data, reward, done, _ = self._env.step_env(key, state.data, action, self._params)
        reward = jnp.asarray(reward, jnp.float32)
        return state.replace(
            data=data, obs=obs.reshape(-1).astype(jnp.float32), reward=reward,
            done=jnp.asarray(done, jnp.float32),
            metrics={**state.metrics, 'reward': reward},
            info={**state.info, 'rng': rng},
        )
