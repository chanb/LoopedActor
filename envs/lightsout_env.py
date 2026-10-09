"""Lights Out environment in JAX (ported from Stoix's stoix/envs/lightsout/lightsout_env.py).

An m x n grid of lights, each on (1) or off (0). Pressing a cell toggles it and its orthogonal
neighbours. The agent presses one cell per step to turn the grid from a random `initial` pattern
into a random `goal` pattern.

Instances are generated procedurally (no level bank): `goal` = `initial` toggled by `dist` distinct
random presses (in GF(2) only the set of pressed cells matters), so it is reachable in at most `dist`
presses. `dist` is drawn per episode from [1, threshold) for training resets (`reset`) and from
[threshold, m*n) for evaluation resets (`eval_reset`), with threshold = int(m*n * difficulty_threshold):
train on short solutions, evaluate on longer ones.

Observation: the current grid (m*n values in {0, 1}, row-major); the goal is the goal vector
(info['target_goal'], goal_size m*n), i.e. the flat grid + goal input mode of the tokenizer
(obs_channels 0: per-cell current/target one-hots, mismatch and geometry).
Actions (per-cell head): a = the cell to press, m*n actions.
Reward (sparse, as in Stoix): 1 when the grid matches the goal (episode ends), 0 otherwise.
Episode length set in make_env / the config (Stoix default 6 for 3x3).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from ml_collections import config_dict

from envs.utils import State


@struct.dataclass
class LightsOutData:
    grid: jax.Array  # [m*n] float32 in {0, 1}
    goal: jax.Array  # [m*n] float32 in {0, 1}
    dist: jax.Array  # () int32 number of presses used to generate the goal (diagnostic only)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        m=3,
        n=3,
        episode_length=6,
        difficulty_threshold=0.5,  # train dist in [1, threshold), eval dist in [threshold, m*n)
        step_penalty=0.0,          # 0 = sparse reward (Stoix)
        solve_reward=1.0,
    )


def press_matrix(m, n):
    """[m*n, m*n] int32: row i = the cells toggled by pressing cell i."""
    mat = np.zeros((m * n, m * n), np.int32)
    for i in range(m * n):
        r, c = divmod(i, n)
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= r + dr < m and 0 <= c + dc < n:
                mat[i, (r + dr) * n + (c + dc)] = 1
    return jnp.asarray(mat)


class LightsOutEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self.m, self.n = config.m, config.n
        self.grid_size = self.m * self.n
        self.obs_channels = 0  # flat grid + goal vector
        self.action_head = 'per_cell'
        self.threshold = int(self.grid_size * config.difficulty_threshold)
        assert 1 < self.threshold < self.grid_size, (self.threshold, self.grid_size)
        self.press = press_matrix(self.m, self.n)

    @property
    def action_size(self): return self.grid_size

    @property
    def observation_size(self): return self.grid_size

    @property
    def goal_size(self): return self.grid_size

    def _make_state(self, rng, data: LightsOutData):
        return State(
            data=data, obs=data.grid,
            reward=jnp.array(0.0, jnp.float32), done=jnp.array(0.0, jnp.float32),
            metrics={'success': 0.0, 'reward': 0.0},
            info={'rng': rng, 'target_goal': data.goal},
        )

    def reset_with_dist(self, rng: jax.Array, dist: jax.Array) -> State:
        """Reset to a random instance whose goal is `dist` distinct presses away from the initial grid."""
        key_init, key_press = jax.random.split(rng)
        initial = jax.random.bernoulli(key_init, shape=(self.grid_size,)).astype(jnp.int32)
        order = jax.random.permutation(key_press, self.grid_size)
        pressed = jnp.zeros(self.grid_size, jnp.int32).at[order].set(jnp.arange(self.grid_size) < dist)
        goal = (initial + pressed @ self.press) % 2
        data = LightsOutData(grid=initial.astype(jnp.float32), goal=goal.astype(jnp.float32),
                             dist=jnp.asarray(dist, jnp.int32))
        return self._make_state(rng, data)

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        return self.reset_with_dist(rng, jax.random.randint(key, (), 1, self.threshold))

    def eval_reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        return self.reset_with_dist(rng, jax.random.randint(key, (), self.threshold, self.grid_size))

    def step(self, state: State, action: jax.Array) -> State:
        d = state.data
        new_grid = ((d.grid.astype(jnp.int32) + self.press[action]) % 2).astype(jnp.float32)
        success = jnp.all(new_grid == d.goal).astype(jnp.float32)
        cfg = self._config
        reward = cfg.solve_reward * success - cfg.step_penalty
        new_data = LightsOutData(grid=new_grid, goal=d.goal, dist=d.dist)
        return State(
            data=new_data, obs=new_grid, reward=reward, done=success,
            metrics={**state.metrics, 'success': success, 'reward': reward},
            info={**state.info, 'target_goal': d.goal},
        )


def render_ascii(grid, goal, m, n):
    """Current grid | goal grid ('#' on, '.' off)."""
    g, t = np.asarray(grid).reshape(m, n), np.asarray(goal).reshape(m, n)
    row = lambda r: ''.join('#' if v else '.' for v in r)
    return '\n'.join(f'{row(a)}  {row(b)}' for a, b in zip(g, t))
