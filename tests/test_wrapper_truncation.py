import jax
import jax.numpy as jnp

from envs.utils import make_env
from utils.wrapper import wrap_env


def test_timeout_is_truncation_after_auto_reset():
    import types
    args = types.SimpleNamespace(env_id='slidingpuzzle-4x4', slidingpuzzle_episode_length=3, slidingpuzzle_num_random_moves=100,
                                 slidingpuzzle_eval_num_random_moves=100)
    cls, cfg = make_env(args)
    env = wrap_env(cls(cfg), 3)
    st = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    for _ in range(3):  # 100-move scrambles are not solved in 3 moves: step 3 is a timeout
        st = jax.jit(env.step)(st, jnp.zeros(4, jnp.int32))
    assert st.done.tolist() == [1.0] * 4
    assert st.info['truncation'].tolist() == [1.0] * 4
