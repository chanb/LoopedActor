import jax
import jax.numpy as jnp
import numpy as np

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
    assert 'final_obs' in st.info and 'final_target_goal' in st.info
    # AutoResetWrapper reset st.obs to a new episode, while final_obs keeps the pre-reset terminal obs.
    assert not jnp.array_equal(st.obs, st.info['final_obs'])


def test_compute_gae_standard_ramdp_and_truncation():
    from ppo import compute_gae
    # 3 steps, 1 env: step 1 truncates with reward 2.0 and true final-state value 10.0; step 2 is a new episode.
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.array([[0.5], [1.0], [0.2]])
    bootstrap_value = jnp.array([0.4])
    truncation = jnp.array([[0.0], [1.0], [0.0]])
    termination = jnp.array([[0.0], [0.0], [0.0]])
    discount = jnp.array([[0.9], [0.8], [0.7]])  # gamma^(C_t)
    trunc_vals = jnp.array([[0.0], [10.0], [0.0]])
    lam = 0.5

    vs, adv = compute_gae(truncation, termination, rewards, values, bootstrap_value,
                          lambda_=lam, discount=discount, truncation_values=trunc_vals)
    # Analytical standard RAMDP GAE:
    # t=2: delta_2 = 3.0 + 0.7 * 0.4 - 0.2 = 3.08; A_2 = 3.08
    # t=1 (truncated): delta_1 = 2.0 + 0.8 * 10.0 - 1.0 = 9.0; A_1 = 9.0 (no leak from t=2)
    # t=0: delta_0 = 1.0 + 0.9 * 1.0 - 0.5 = 1.4; A_0 = 1.4 + 0.9 * 0.5 * 9.0 = 5.45
    want_adv = np.array([[5.45], [9.0], [3.08]], dtype=np.float32)
    want_vs = want_adv + np.asarray(values)
    assert np.allclose(np.asarray(adv), want_adv, atol=1e-5)
    assert np.allclose(np.asarray(vs), want_vs, atol=1e-5)

