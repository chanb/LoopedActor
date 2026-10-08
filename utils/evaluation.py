import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from utils.wrapper import EvalWrapper


def generate_unroll(env, env_state, policy, key, unroll_length,
                    discount_act=1.0, discount_compute=1.0):
    """Step the env for unroll_length steps.

    Returns the final state and the per-env discounted returns of the first
    episode, as a dict:
      'discounted_return'         sum_t discount_act^t r_t (discounting per env step)
      'compute_discounted_return' sum_t discount_act^t discount_compute^(z_t + c_t - 1) r_t,
                                  z_t = sum_{k<t} (c_k - 1), with c_t the policy's compute time
                                  (the return PPO optimizes; equal to the above when discount_compute = 1)
    """
    def body(i, carry):
        env_state, key, returns, discounts = carry
        key, next_key = jax.random.split(key)
        actions, policy_extras = policy(env_state.obs, env_state.info["target_goal"], key)
        next_env_state = env.step(env_state, actions)

        # Only the first episode counts (EvalWrapper's active flag before this step).
        active = env_state.info['eval_metrics'].active_episodes
        compute_time = policy_extras['compute_time'].astype(jnp.float32)
        compute_discount = discount_compute ** (compute_time - 1)
        step_discounts = {
            'discounted_return': (1.0, discount_act),
            'compute_discounted_return': (compute_discount, discount_act * compute_discount),
        }
        for name, (reward_discount, next_discount) in step_discounts.items():
            returns[name] = returns[name] + active * discounts[name] * reward_discount * next_env_state.reward
            discounts[name] = discounts[name] * next_discount
        return (next_env_state, next_key, returns, discounts)

    zeros = jnp.zeros_like(env_state.reward)
    names = ('discounted_return', 'compute_discounted_return')
    final_state, _, returns, _ = jax.lax.fori_loop(
        0, unroll_length, body,
        (env_state, key, {k: zeros for k in names}, {k: zeros + 1.0 for k in names}),
    )
    return final_state, returns


class Evaluator:
    def __init__(
        self,
        eval_env,
        eval_policy_fn: Callable,
        num_eval_envs: int,
        episode_length: int,
        key: jax.Array,
        discount_act: float = 1.0,
        discount_compute: float = 1.0,
    ):
        self._key = key
        self._eval_walltime = 0.0
        # Wrap the environment with EvalWrapper for metric tracking
        eval_env = EvalWrapper(eval_env)

        # Helper to define the unroll logic for either reset or eval_reset
        def make_unroll_fn(reset_type: str):
            def generate_unroll_logic(policy_params, key):
                reset_keys = jax.random.split(key, num_eval_envs)
                reset_fn = getattr(eval_env, reset_type)
                eval_first_state = reset_fn(reset_keys)
                return generate_unroll(
                    eval_env,
                    eval_first_state,
                    eval_policy_fn(policy_params),
                    key,
                    unroll_length=episode_length,
                    discount_act=discount_act,
                    discount_compute=discount_compute,
                )
            return jax.jit(generate_unroll_logic)

        # Two separate JIT'd functions: in-distribution ("easy" resets) and
        # the held-out harder distances ("hard" eval resets).
        self._generate_train_unroll = make_unroll_fn("reset")
        self._generate_test_unroll = make_unroll_fn("eval_reset")

        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics,
    ):
        """Run evaluation on both the train and test difficulty splits."""
        self._key, train_key, test_key = jax.random.split(self._key, 3)

        t = time.time()

        # 1. Trajectories for the train split (in-distribution)
        train_state, train_returns = self._generate_train_unroll(policy_params, train_key)
        # 2. Trajectories for the test split (out-of-distribution)
        test_state, test_returns = self._generate_test_unroll(policy_params, test_key)

        # Block to ensure execution finishes for timing
        test_state.info['eval_metrics'].active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t

        metrics = {}

        for split_name, state, returns in [("train", train_state, train_returns), ("test", test_state, test_returns)]:
            eval_metrics = state.info['eval_metrics']

            for name, value in eval_metrics.episode_metrics.items():
                metrics[f'eval/{split_name}_episode_{name}'] = np.mean(value)

                if name == "success":
                    metrics[f'eval/{split_name}_episode_{name}_rate'] = np.mean(value > 0)

            metrics[f'eval/{split_name}_avg_episode_length'] = np.mean(eval_metrics.episode_steps)

            for name, value in returns.items():
                metrics[f'eval/{split_name}_episode_{name}'] = np.mean(value)

        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = (self._steps_per_unroll * 2) / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        metrics['eval/walltime'] = self._eval_walltime

        return {**training_metrics, **metrics}
