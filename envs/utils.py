import re
from typing import Any, Dict

import jax
from flax import struct


@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: Any
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


def make_env(args):
    """Build the environment class and config from an env id
    (sokoban-<train>-<eval> | rushhour-<train>-<eval> | lightsout-<m>x<n> | slidingpuzzle-<N>x<N>)."""
    sok = re.fullmatch(r"sokoban-([a-z_]+)-([a-z_]+)", args.env_id)  # sokoban-<train_split>-<eval_split>
    if sok is not None:
        from envs.sokoban_env import SokobanEnv, default_config as sokoban_config
        config = sokoban_config()
        config.train_split, config.eval_split = sok.group(1), sok.group(2)
        config.max_train_levels = getattr(args, 'sokoban_max_train_levels', 0)
        config.episode_length = getattr(args, 'sokoban_episode_length', 120)
        return SokobanEnv, config
    rh = re.fullmatch(r"rushhour-([a-z_]+)-([a-z_]+)", args.env_id)  # rushhour-<train_split>-<eval_split>
    if rh is not None:
        from envs.rushhour_env import RushHourEnv, default_config as rushhour_config
        config = rushhour_config()
        config.train_split, config.eval_split = rh.group(1), rh.group(2)
        config.max_train_levels = getattr(args, 'rushhour_max_train_levels', 0)
        config.episode_length = getattr(args, 'rushhour_episode_length', 150)
        config.shaping_weight = getattr(args, 'rushhour_shaping_weight', 1.0)
        return RushHourEnv, config
    lo = re.fullmatch(r"lightsout-(\d+)x(\d+)", args.env_id)  # lightsout-<m>x<n>, procedurally generated
    if lo is not None:
        from envs.lightsout_env import LightsOutEnv, default_config as lightsout_config
        config = lightsout_config()
        config.m, config.n = int(lo.group(1)), int(lo.group(2))
        config.episode_length = getattr(args, 'lightsout_episode_length', 6)
        config.difficulty_threshold = getattr(args, 'lightsout_difficulty_threshold', 0.5)
        return LightsOutEnv, config
    sp = re.fullmatch(r"slidingpuzzle-(\d+)x(\d+)", args.env_id)  # slidingpuzzle-<N>x<N>, procedurally generated
    if sp is not None:
        if sp.group(1) != sp.group(2):
            raise ValueError(f"slidingpuzzle needs a square board, got {args.env_id}")
        from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config as slidingpuzzle_config
        config = slidingpuzzle_config()
        config.grid_size = int(sp.group(1))
        config.episode_length = getattr(args, 'slidingpuzzle_episode_length', 40)
        config.num_random_moves = getattr(args, 'slidingpuzzle_num_random_moves', 10)
        eval_moves = getattr(args, 'slidingpuzzle_eval_num_random_moves', 0)
        config.eval_num_random_moves = eval_moves if eval_moves > 0 else config.num_random_moves
        return SlidingPuzzleEnv, config
    raise ValueError(f"Environment {args.env_id} not supported (expected sokoban-<train>-<eval>, "
                     f"rushhour-<train>-<eval>, lightsout-<m>x<n> or slidingpuzzle-<N>x<N>)")
