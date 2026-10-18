"""PPO training for a Fixed-Point Reasoning Model (FPRM) actor-critic on
grid puzzles (Boxoban, Rush Hour), with adaptive per-state thinking via a policy-KL halting criterion.

The actor-critic is a weight-tied looped Transformer iterated by a
fixed-point solver ("thinking"). At every state, the network stops iterating
once the KL divergence between consecutive policy readouts drops below
`--halt_kl` — it thinks only as long as thinking still changes its mind. The
PPO loss is applied to the final actor/value readout, with full
backpropagation through every executed think iteration.

Run:
    python ppo.py --env_id=sokoban-unfiltered_train-unfiltered_valid --architecture=fprm --seed=1
"""

import os
import time
import pickle
import pprint
import functools

import numpy as np
import tyro

import jax
import jax.numpy as jnp
import flax
import optax
import distrax

from pathlib import Path
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional
from flax.training.train_state import TrainState

import utils.running_statistics as running_statistics
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from envs.utils import make_env
from models.fprm import FPRMConfig
from models.fprm_thinker import FPRMThinkerActorValue
from models.fprm_cot_thinker import FPRMCoTThinkerActorValue
from models.fprm_perceiver import FPRMPerceiverActorValue
from models.fprm_mem import FPRMMemActorValue
from models.fprm_mem_halt import FPRMMemHaltActorValue
from models.fprm_rim import FPRMRiMActorValue
from models.transformer_baseline import TransformerActorValue


@dataclass
class Args:
    # experiment
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    # logging and checkpointing
    track: bool = False                   # log to wandb
    wandb_project_name: str = "looped_actor"
    wandb_entity: Optional[str] = None
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50              # number of evaluation / logging / saving steps
    num_reset_steps: int = 50             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'sokoban-unfiltered_train-unfiltered_valid'  # sokoban-<train>-<eval> | rushhour-<train>-<eval> | lightsout-<m>x<n> | slidingpuzzle-<N>x<N>
    num_envs: int = 2048
    num_eval_envs: int = 128
    sokoban_max_train_levels: int = 0     # sokoban-*: cap on training levels (0 = whole split)
    sokoban_episode_length: int = 120     # sokoban-*: steps per episode
    rushhour_max_train_levels: int = 0    # rushhour-*: cap on training levels (0 = whole split)
    rushhour_episode_length: int = 150    # rushhour-*: steps per episode
    rushhour_shaping_weight: float = 1.0  # rushhour-*: potential-based shaping weight (0 = sparse solve reward only)
    lightsout_episode_length: int = 6     # lightsout-*: steps per episode (training resets and eval/train_* episodes)
    lightsout_eval_episode_length: int = 0  # lightsout-*: steps per held-out (eval/test_*) episode; 0 = lightsout_episode_length
    lightsout_difficulty_threshold: float = 0.5  # lightsout-*: train goals need < int(m*n*this) presses, eval goals >= it
    slidingpuzzle_episode_length: int = 40        # slidingpuzzle-*: steps per episode (training resets and eval/train_* episodes)
    slidingpuzzle_eval_episode_length: int = 0    # slidingpuzzle-*: steps per evaluation-scramble (eval/test_*) episode;
    slidingpuzzle_num_random_moves: int = 10      # slidingpuzzle-*: training scramble length (random walk of the blank)
    slidingpuzzle_eval_num_random_moves: int = 0  # slidingpuzzle-*: evaluation scramble length (0 = same as training)

    # algorithm
    num_timesteps: int = 50000000
    rollout_length: int = 160
    num_minibatches_per_rollout: int = 32
    num_epochs_per_rollout: int = 8
    learning_rate: float = 1e-4
    # Return G = sum_t discount_act^t * discount_compute^(z_t + c_t - 1) * r_t with z_t = sum_{k<t} (c_k - 1),
    # where c_t is the compute time (think iterations) of decision t. discount_compute = 1 is the
    # standard per-env-step return; discount_compute = discount_act is the RAMDP return.
    discount_act: float = 0.99
    discount_compute: float = 1.0
    entropy_cost: float = 0.01
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    value_at_max_iters: bool = False      # fprm_mem: the value is read out from the max_think_iters-th pass's latent (halting
                                          # fixes only the policy readout; the latent keeps evolving), not from the halt pass
    value_expectile: float = 0.5          # value loss |tau - 1(u < 0)| u^2 with u = target - V: 0.5 = MSE (mean return),
                                          # > 0.5 = an upper expectile of the return (optimistic V, as Stoix --expectile)
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    normalize_observations: bool = False  # running mean/std normalization of the observations (per input feature, so tied to the
                                          # board size); off: the models see raw observations and checkpoints store an identity normalizer
    # stabilization (all off by default)
    max_grad_norm: float = 0.0            # 0 = no gradient clipping
    target_kl: float = 0.0                # 0 = no KL early stop; else skip remaining minibatch updates in a rollout once approx_kl exceeds this
    anneal_lr: bool = False               # linearly decay lr to 0 over training

    # model
    # 'fprm'         — looped FPRM with policy-KL adaptive halting
    # 'fprm_cot'     — FPRM thinking by appending continuous thought tokens (implicit CoT, causal, KV-cached); policy-KL halting only
    # 'fprm_perceiver' — Perceiver AR-style latent CoT: latents (from <BOT>) attend to the cell tokens + earlier latents at every layer
    # 'fprm_mem'       — fprm's looped board plus a memory of thought tokens (cells and thoughts attend to each other; readout from the newest thought)
    # 'fprm_mem_halt'  — fprm_mem with stochastic halting: c ~ p(c|s) from h_i = sigmoid(beta - alpha log KL_i) (learned alpha, beta);
    #                    the PPO ratio uses the joint log p(c|s) + log pi_c(a|s); halt_kl only initializes beta
    # 'fprm_rim'       — RiM-style: each step appends [grid-conv'd cell output latents, SEP] (block-causal, KV-cached); halts when the
    #                    SEP latent converges (--halt_criterion latent_residual) or by policy KL (kl)
    # 'single_block' — FPRM core applied once (same params as fprm, no looping)
    # 'multi_block'  — num_blocks FPRM cores without weight tying (~num_blocks x params)
    architecture: str = 'fprm'
    num_blocks: int = 8                   # blocks of the multi_block baseline
    # observation tokenization (all architectures): 'cell' = one token per grid cell from its channels +
    # geometry; 'flat' = one token per observation dimension; 'conv' = 3x3 conv, one token per output channel
    tokenizer: str = 'cell'
    tokenizer_conv_features: int = 64     # tokenizer='conv': conv output channels (= number of tokens)
    residual_scaling: bool = True         # --no-residual_scaling: plain pre-norm residuals in the FPRM core (no a1/b1 block scaling, no a2/b2 input mixing) — only sensible for non-looped baselines (no contractivity needed)
    mlp_type: str = 'swiglu'              # block MLP: 'swiglu' (default) | 'gelu'
    mlp_round_to: int = 256               # MLP hidden width rounded up to a multiple of this (256 = default; at d=128: swiglu 512, 64 -> 384, 1 -> 341)
    compute_dtype: str = 'float32'        # block computation dtype: 'float32' | 'float16' | 'bfloat16' (params, carried latent, heads stay float32);
                                          # bfloat16 is the safe choice: float16 activations can overflow (inf -> NaN), which loss scaling cannot fix
    dynamic_loss_scale: bool = True       # compute_dtype=float16 only: scale the loss by S so small fp16 gradients do not underflow; a minibatch
                                          # with a nonfinite gradient is skipped and S halved, S doubles after loss_scale_period good steps
    loss_scale_init: float = 4096.0       # initial S (fp16 gradients underflow at S = 1 and overflow at S = 2^16 at initialization)
    loss_scale_period: int = 1000         # good minibatch steps before S is doubled

    # FPRM thinker
    fprm_d_model: int = 128
    fprm_num_heads: int = 4
    fprm_num_layers: int = 1
    max_think_iters: int = 8              # cap on core calls ("think iterations") per decision
    min_think_iters: int = 2              # halting is checked only after this many iterations
    halt_criterion: str = 'kl'            # 'kl' (policy-KL halting) | 'latent_residual' (original FPRM-paper rule: latent max-token residual)
    halt_kl: float = 1e-3                 # stop thinking once KL(pi_i || pi_{i-1}) between consecutive policy readouts <= this
                                          # (0 = only at a fixed point, < 0 = never: the iso_* arms pass -1)
    halt_residual_thresh: float = 0.1     # halt_criterion='latent_residual': stop once the latent residual < this (paper fp_thresh;
                                          # fprm_rim: relative change of the SEP latent between steps)
    perceiver_step_grid_conv: bool = False  # fprm_perceiver: apply a residual depthwise 3x3 conv to the cells at every thinking step
                                            # (pass i reads the cells convolved i times; tokenizer=cell)
    perceiver_num_initial_latents: int = 1  # fprm_perceiver: number of learned initial latents (bidirectional among themselves, all in pass 1)
    perceiver_cell_encoder_iters: int = 0   # fprm_perceiver: looped board encoder iterations (0 = off; latent layer l reads the last
                                            # iteration's layer-l output)
    input_grid_conv: bool = False         # fprm_cot / fprm_perceiver: residual depthwise 3x3 conv over the cell embeddings (tokenizer=cell)
    input_grid_conv_depth: int = 1        # fprm_cot / fprm_perceiver with --input_grid_conv: 1 = one linear depthwise conv; D > 1 = D ConvNeXt blocks
    halt_beta_head: bool = False          # fprm_mem_halt: state-dependent beta (Dense on the newest thought) instead of a scalar
    halt_head_grad: bool = False          # fprm_mem_halt: let the halting loss backpropagate into the trunk through the beta head
    halt_entropy_cost: float = 0.0        # fprm_mem_halt: entropy bonus on p(c|s) (the action entropy keeps entropy_cost)
    halt_update: str = 'joint'            # fprm_mem_halt policy update: 'joint' = PPO ratio of the sampled (c, a), log p(c|s) + log pi_c(a|s);
                                          # 'all_c' = every think time c from the one sampled action, using Q(s, c, a) = g_c^(c-1) Q_env(s, a)
                                          # (see compute_ppo_loss);
                                          # 'tpo' = Target Policy Optimization (Kaddour, 2026) on p(c|s): cross-entropy to the frozen target
                                          # q(c|s) ∝ p_old(c|s) exp(u_c / eta), u_c = g_c^(c-1) (V + pi_old,c(a|s) / pi_bar(a|s) (Q_env - V)),
                                          # plus PPO on the sampled c's readout pi_c(a|s) (see learn_step)
    halt_tpo_eta: float = 1.0             # halt_update='tpo': target temperature, in units of the batch advantage std
    halt_explore: float = 0.0             # fprm_mem_halt, halt_update='all_c'|'tpo': rollouts sample c from (1 - this) p(c|s) + this * Uniform
    halt_extend: int = 0                  # fprm_mem_halt, halt_update='tpo': the action loss trains every readout pi_c', c' <= c + this
                                          # (c the sampled think time; >= max_think_iters trains every pass), not only pi_c
    cot_thought_norm: bool = False        # fprm_cot / fprm_perceiver / fprm_mem: RMS-normalize each thought / latent before feeding it back


@flax.struct.dataclass
class PPOTrainingState(TrainState):
    """Contains training state for the learner."""
    normalizer_params: Any
    env_steps: float
    loss_scale: Any  # () float32, dynamic fp16 loss scale (unused unless compute_dtype=float16 and dynamic_loss_scale)
    loss_scale_good_steps: Any  # () int32, consecutive finite-gradient steps since the last scale change


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()


def count_parameters(params):
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))


def save_params(path, params):
    with open(path, 'wb') as fout:
        fout.write(pickle.dumps(params))


def load_params(path):
    with open(path, 'rb') as fin:
        return pickle.loads(fin.read())


def make_inference_fn(actor_critic_network, stochastic_halt: bool = False, halt_all_c: bool = False):
    """Creates params and inference function for the PPO agent.

    stochastic_halt: the network samples its think time c (fprm_mem_halt); the stored log_prob is the joint
    log p(c|s) + log pi_c(a|s), and compute_time = c is replayed in the loss. Deterministic: c = argmax p(c|s).
    halt_all_c: also store, for every think time c, log p(c|s) and log pi_c(a|s) of the sampled action, and the
    behaviour marginal log pi_bar(a|s) = log sum_c mu(c|s) pi_c(a|s) (c is sampled from mu, see halt_explore),
    and log mu(c|s) (halt_update='tpo' reweights the sampled c's action update by p_old(c|s) / mu(c|s)).
    """
    def make_policy(params, deterministic: bool = False):

        def policy(observations, goals, key_sample):
            inputs = jnp.concatenate([observations, goals], axis=-1)
            halt_kw = {}
            if stochastic_halt and not deterministic:
                key_sample, key_halt = jax.random.split(key_sample)
                halt_kw = {'halt_key': key_halt}
            (policy_dist, value), inter = actor_critic_network.apply(
                params['policy'],
                inputs,
                params['normalizer'],
                mutable=['intermediates'],
                **halt_kw,
            )
            # Compute time of this decision = number of think iterations (core calls).
            compute_time = inter['intermediates']['fp_iterations'][0]

            if deterministic:
                return policy_dist.mode(), {'value': value, 'compute_time': compute_time}

            actions = policy_dist.sample(seed=key_sample)
            log_prob = policy_dist.log_prob(actions)
            if stochastic_halt:
                log_prob = log_prob + inter['intermediates']['halt_log_prob'][0]
            extras = {
                'log_prob': log_prob,
                'value': value,
                'compute_time': compute_time,
            }
            if halt_all_c:
                I = inter['intermediates']
                log_pa = jnp.take_along_axis(
                    jax.nn.log_softmax(I['logits_all'][0], axis=-1), actions[..., None, None], axis=-1)[..., 0]  # [B, T]
                extras.update({
                    'halt_old_log_pc': I['halt_log_probs'][0],
                    'halt_old_log_pa': log_pa,
                    'halt_log_mu_a': jax.nn.logsumexp(I['halt_behaviour_log_probs'][0] + log_pa, axis=-1),
                    'halt_old_log_mu': I['halt_behaviour_log_probs'][0],
                })
            return actions, extras

        return policy
    return make_policy


def make_optimizer(args):
    if args.anneal_lr:
        total_grad_steps = (
            args.num_training_step
            * args.num_epochs_per_rollout
            * args.num_minibatches_per_rollout
        )
        learning_rate = optax.linear_schedule(
            args.learning_rate, 0.0, total_grad_steps
        )
    else:
        learning_rate = args.learning_rate
    tx = optax.adam(learning_rate=learning_rate)
    if args.max_grad_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), tx)
    return tx


def main(args: Args):

    assert 0.0 < args.value_expectile < 1.0, args.value_expectile
    assert not args.value_at_max_iters or args.architecture == 'fprm_mem', '--value_at_max_iters is implemented for fprm_mem only'
    args.num_training_step = args.num_timesteps // (args.num_envs * args.rollout_length)
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    args.minibatch_size = args.num_envs * args.rollout_length // (args.num_minibatches_per_rollout)

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.architecture}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, key_env, key_eval, key_policy = jax.random.split(key, 4)

    # Initialize environment
    env_class, env_config = make_env(args)
    env = wrap_env(env_class(config=env_config), env_config.episode_length)
    eval_env = wrap_env(env_class(config=env_config), env_config.episode_length, train=False)
    episode_length = env_config.episode_length
    # Held-out evaluation episodes may get a longer budget (lightsout: their goals need more presses).
    test_episode_length = env_config.get('eval_episode_length', episode_length)
    test_eval_env = (wrap_env(env_class(config=env_config), test_episode_length, train=False)
                     if test_episode_length != episode_length else eval_env)

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size
    obs_channels = getattr(env, 'obs_channels', 0)          # > 0 = plane-tokenized grid observation (Boxoban, Rush Hour)
    action_head = getattr(env, 'action_head', 'readout' if obs_channels > 0 else 'per_cell')  # plane-tokenized envs use the readout head by default

    # Initialize the actor-critic
    fprm_config = FPRMConfig(
        d_model=args.fprm_d_model,
        num_heads=args.fprm_num_heads,
        num_layers=args.fprm_num_layers,
        residual_scaling=args.residual_scaling,
        mlp_type=args.mlp_type,
        mlp_round_to=args.mlp_round_to,
        compute_dtype=args.compute_dtype,
    )
    if args.architecture == 'fprm':
        actor_critic_network = FPRMThinkerActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_criterion=args.halt_criterion,
            halt_kl=args.halt_kl,
            halt_residual_thresh=args.halt_residual_thresh,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
        )
    elif args.architecture == 'fprm_cot':
        assert args.halt_criterion == 'kl', "fprm_cot supports only halt_criterion='kl'"
        actor_critic_network = FPRMCoTThinkerActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_kl=args.halt_kl,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
            thought_norm=args.cot_thought_norm,
            input_grid_conv=args.input_grid_conv,
            input_grid_conv_depth=args.input_grid_conv_depth,
        )
    elif args.architecture == 'fprm_perceiver':
        assert args.halt_criterion == 'kl', "fprm_perceiver supports only halt_criterion='kl'"
        actor_critic_network = FPRMPerceiverActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_kl=args.halt_kl,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
            thought_norm=args.cot_thought_norm,
            input_grid_conv=args.input_grid_conv,
            input_grid_conv_depth=args.input_grid_conv_depth,
            step_grid_conv=args.perceiver_step_grid_conv,
            num_initial_latents=args.perceiver_num_initial_latents,
            cell_encoder_iters=args.perceiver_cell_encoder_iters,
        )
    elif args.architecture == 'fprm_mem':
        assert args.halt_criterion == 'kl', "fprm_mem supports only halt_criterion='kl'"
        actor_critic_network = FPRMMemActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_kl=args.halt_kl,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
            thought_norm=args.cot_thought_norm,
            value_at_max_iters=args.value_at_max_iters,
        )
    elif args.architecture == 'fprm_mem_halt':
        assert args.halt_criterion == 'kl', "fprm_mem_halt supports only halt_criterion='kl'"
        actor_critic_network = FPRMMemHaltActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_kl=args.halt_kl,
            halt_beta_head=args.halt_beta_head,
            halt_head_grad=args.halt_head_grad,
            halt_explore=args.halt_explore,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
            thought_norm=args.cot_thought_norm,
        )
    elif args.architecture == 'fprm_rim':
        assert args.tokenizer == 'cell', "fprm_rim needs tokenizer='cell'"
        actor_critic_network = FPRMRiMActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_criterion='latent' if args.halt_criterion == 'latent_residual' else 'kl',
            halt_residual_thresh=args.halt_residual_thresh,
            halt_kl=args.halt_kl,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
        )
    elif args.architecture in ('single_block', 'multi_block'):
        actor_critic_network = TransformerActorValue(
            m=env.m,
            n=env.n,
            config=fprm_config,
            num_blocks=1 if args.architecture == 'single_block' else args.num_blocks,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    training_state = PPOTrainingState.create(
        apply_fn=None,
        params=actor_critic_network.init(key_policy, x=jnp.zeros((1, obs_size + goal_size))),
        tx=make_optimizer(args),
        normalizer_params=running_statistics.init_state((obs_size + goal_size,)),
        env_steps=np.zeros((), dtype=np.float64),
        loss_scale=jnp.asarray(args.loss_scale_init, jnp.float32),
        loss_scale_good_steps=jnp.zeros((), jnp.int32),
    )
    stochastic_halt = args.architecture == 'fprm_mem_halt'
    assert args.halt_update in ('joint', 'all_c', 'tpo'), args.halt_update
    halt_all_c = stochastic_halt and args.halt_update == 'all_c'
    halt_tpo = stochastic_halt and args.halt_update == 'tpo'
    assert args.halt_tpo_eta > 0, args.halt_tpo_eta
    # 'joint' stores the target's log p(c|s) as the behaviour probability, so c must be sampled from p itself.
    assert halt_all_c or halt_tpo or args.halt_explore == 0.0, \
        '--halt_explore needs --architecture=fprm_mem_halt --halt_update=all_c|tpo'
    assert halt_tpo or args.halt_extend == 0, '--halt_extend needs --architecture=fprm_mem_halt --halt_update=tpo'
    make_policy = make_inference_fn(actor_critic_network, stochastic_halt, halt_all_c or halt_tpo)

    print(f'\nNumber of parameters in actor critic network are: {count_parameters(training_state.params)}\n')

    # Initialize evaluator
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        test_eval_env=test_eval_env,
        test_episode_length=test_episode_length,
        key=key_eval,
        discount_act=args.discount_act,
        discount_compute=args.discount_compute,
    )

    def generate_unroll(env, env_state, policy, key, unroll_length, extra_fields):
        """Collect trajectories of given unroll_length."""
        @jax.jit
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            actions, policy_extras = policy(env_state.obs, env_state.info['target_goal'], key)

            next_env_state = env.step(env_state, actions)
            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            transition = Transition(
                observation=jnp.concatenate([env_state.obs, env_state.info['target_goal']], axis=-1),
                action=actions,
                value=policy_extras['value'],
                reward=next_env_state.reward,
                discount=1 - next_env_state.done,
                next_observation=jnp.concatenate([next_env_state.obs, next_env_state.info['target_goal']], axis=-1),
                extras={'policy_extras': policy_extras, 'state_extras': state_extras},
            )

            return (next_env_state, next_key), transition

        (final_env_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
        return final_env_state, data

    def obs_normalizer(training_state):
        """The normalizer passed to the network (None = raw observations)."""
        return training_state.normalizer_params if args.normalize_observations else None

    @jax.jit
    def data_collect_step(training_state, env_state, key_generate_rollout):
        policy = make_policy({
            'policy': training_state.params,
            'normalizer': obs_normalizer(training_state),
        })

        env_state, data = generate_unroll(
            env,
            env_state,
            policy,
            key_generate_rollout,
            args.rollout_length,
            extra_fields=('truncation',),
        )

        # Update normalization params (left at the identity init when observations are not normalized).
        if args.normalize_observations:
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
            )
        else:
            normalizer_params = training_state.normalizer_params

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )

        return training_state, env_state, data

    def compute_gae(
        truncation: jnp.ndarray,
        termination: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        bootstrap_value: jnp.ndarray,
        lambda_: float = 1.0,
        discount=0.99,
    ):
        # `discount` is a scalar or a per-step [T, B] array (discount_act * discount_compute^(c_t - 1)).
        discount = jnp.broadcast_to(discount, termination.shape)
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)

        def compute_vs_minus_v_xs(carry, target_t):
            lambda_, acc = carry
            truncation_mask, delta, termination, discount = target_t
            acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), (acc)

        (_, _), (vs_minus_v_xs) = jax.lax.scan(
            compute_vs_minus_v_xs,
            (lambda_, acc),
            (truncation_mask, deltas, termination, discount),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        advantages = (
            rewards + discount * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)

    def compute_ppo_loss(params, normalizer_params, data, rng):
        data, value_targets, advantages = data

        def ppo_terms(policy_dist, baseline, halt_log_prob=0.0,
                      behaviour_action_log_probs=data.extras['policy_extras']['log_prob'], sample_weight=1.0):
            # Policy function loss (stochastic halting: joint log p(c|s) + log pi_c(a|s))
            target_action_log_probs = policy_dist.log_prob(data.action) + halt_log_prob
            rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
            surrogate_loss1 = rho_s * advantages
            surrogate_loss2 = (jnp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
            policy_loss = -jnp.mean(sample_weight * jnp.minimum(surrogate_loss1, surrogate_loss2))

            # Value function loss: expectile regression (tau = 0.5 gives the MSE 0.25 * mean(v_error^2)).
            v_error = value_targets - baseline
            v_weight = jnp.abs(args.value_expectile - (v_error < 0).astype(v_error.dtype))
            v_loss = jnp.mean(v_weight * v_error * v_error) * 0.5

            # Entropy loss
            entropy = jnp.mean(policy_dist.entropy())
            entropy_loss = args.entropy_cost * -entropy

            # k3 estimator of KL(behaviour || target), Schulman (2020).
            log_rho = target_action_log_probs - behaviour_action_log_probs
            approx_kl = jax.lax.stop_gradient(
                jnp.mean(jnp.exp(log_rho) - 1.0 - log_rho)
            )
            return policy_loss, v_loss, entropy_loss, approx_kl

        def all_c_terms(I):
            """halt_update='all_c': the policy loss over every think time c from the one sampled action a.

            Since Q(s, c, a) = g_c^(c-1) Q_env(s, a), `advantages` holds A_c = g_c^(c-1) Q_env(s, a) - V(s) for every c
            ([B, T], see learn_step). With the behaviour marginal pi_bar(a|s) = sum_c mu(c|s) pi_old,c(a|s),
              loss = -sum_c w_c min(r_c A_c, clip(r_c) A_c),  w_c = p_old(c|s) pi_old,c(a|s) / pi_bar(a|s),
              r_c = p(c|s) pi_c(a|s) / (p_old(c|s) pi_old,c(a|s)),
            the PPO version of the unbiased estimator sum_c grad(p(c|s) pi_c(a|s)) / pi_bar(a|s) A_c. Think times
            that cannot halt (p forced to 0) get no weight. Returns (policy_loss, entropy, approx_kl, metrics).
            """
            ex = data.extras['policy_extras']
            old_lpc, old_lpa, log_mu_a = ex['halt_old_log_pc'], ex['halt_old_log_pa'], ex['halt_log_mu_a']
            log_pi_all = jax.nn.log_softmax(I['logits_all'][0], axis=-1)  # [B, T, A]
            lpa = jnp.take_along_axis(log_pi_all, data.action[:, None, None].astype(jnp.int32), axis=-1)[..., 0]
            lpc = I['halt_log_probs'][0]
            valid = old_lpc > -1e8  # forced-zero think times have log p = -1e9 (models/fprm_mem_halt.py)
            log_r = jnp.where(valid, (lpc + lpa) - (old_lpc + old_lpa), 0.0)
            w = jnp.where(valid, jnp.exp(old_lpc + old_lpa - log_mu_a[:, None]), 0.0)
            r = jnp.exp(log_r)
            surr = jnp.minimum(r * advantages, jnp.clip(r, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
            policy_loss = -jnp.mean(jnp.sum(w * surr, axis=-1))
            # Action entropy of every pass, weighted by p(c|s) (no gradient into p: the bonus must not favour think
            # times whose readout happens to be more random).
            ent_all = -jnp.sum(jnp.exp(log_pi_all) * log_pi_all, axis=-1)  # [B, T]
            entropy = jnp.mean(jnp.sum(jax.lax.stop_gradient(jnp.exp(lpc)) * ent_all, axis=-1))
            approx_kl = jax.lax.stop_gradient(jnp.mean(jnp.sum(w * (r - 1.0 - log_r), axis=-1)))
            T = lpc.shape[-1]
            metrics = {
                'halt_weight_sum': jnp.mean(jnp.sum(w, axis=-1)),  # ~1 (exactly 1 at halt_explore = 0)
                'halt_expected_c': jnp.mean(jnp.sum(jnp.exp(lpc) * jnp.arange(1, T + 1), axis=-1)),
            }
            return policy_loss, entropy, approx_kl, metrics

        # PPO loss on the final actor / value readout. Halted examples freeze
        # inside the network, so their readout is the one they halted at.
        # Stochastic halting: replay the rollout's think time c.
        halt_kw = {'halt_steps': data.extras['policy_extras']['compute_time']} if stochastic_halt else {}
        _, inter = actor_critic_network.apply(
            params, data.observation, normalizer_params,
            mutable=['intermediates'],
            **halt_kw,
        )
        logits = inter['intermediates']['logits'][0]
        values = inter['intermediates']['value'][0]
        halt_metrics = {}
        if stochastic_halt:
            halt_log_prob = inter['intermediates']['halt_log_prob'][0]
            halt_entropy = jnp.mean(inter['intermediates']['halt_entropy'][0])
            halt_metrics = {
                'halt_entropy': halt_entropy,
                'halt_alpha': inter['intermediates']['halt_alpha'][0],
                'halt_beta': jnp.mean(inter['intermediates']['halt_beta_t'][0]),
            }
            # Mean h_i per pass (forced 0 before min_think_iters, 1 at the cap).
            mean_h = jnp.mean(inter['intermediates']['halt_prob'][0], axis=-1)
            halt_metrics.update({f'halt_prob_{i + 1}': mean_h[i] for i in range(mean_h.shape[0])})
            # Latent residual between consecutive passes: at the sampled (halted) c, and its mean at every pass.
            res_t = inter['intermediates']['latent_residual_t'][0]  # [T, B]
            c_idx = (data.extras['policy_extras']['compute_time'] - 1).astype(jnp.int32)
            halt_metrics['latent_res_at_c'] = jnp.mean(jnp.take_along_axis(res_t, c_idx[None], axis=0))
            mean_res = jnp.mean(res_t, axis=-1)
            halt_metrics.update({f'latent_res_{i + 1}': mean_res[i] for i in range(mean_res.shape[0])})
        else:
            halt_log_prob, halt_entropy = 0.0, 0.0
        if halt_all_c:
            policy_loss, entropy, approx_kl, all_c_metrics = all_c_terms(inter['intermediates'])
            halt_metrics.update(all_c_metrics)
            v_error = value_targets - values
            v_weight = jnp.abs(args.value_expectile - (v_error < 0).astype(v_error.dtype))
            v_loss = jnp.mean(v_weight * v_error * v_error) * 0.5
            entropy_loss = args.entropy_cost * -entropy
        elif halt_tpo:
            # Action head: PPO on the sampled c's readout pi_c(a|s) alone (no log p(c|s) in the ratio), reweighted by
            # p_old(c|s) / mu(c|s) so that halt_explore does not shift which readouts the action head trains on.
            ex = data.extras['policy_extras']
            c_idx = (ex['compute_time'] - 1).astype(jnp.int32)[:, None]
            old_lpa_c = jnp.take_along_axis(ex['halt_old_log_pa'], c_idx, axis=-1)[:, 0]
            is_c = jnp.exp(jnp.take_along_axis(ex['halt_old_log_pc'] - ex['halt_old_log_mu'], c_idx, axis=-1)[:, 0])
            policy_loss, v_loss, entropy_loss, approx_kl = ppo_terms(
                distrax.Categorical(logits=logits), values,
                behaviour_action_log_probs=old_lpa_c, sample_weight=jax.lax.stop_gradient(is_c),
            )
            if args.halt_extend > 0:
                # Train every readout pi_c', c' <= c + halt_extend, not only the sampled c's, so that u_c reflects what each
                # depth can achieve rather than how much each has been trained. PPO with the per-pass ratio
                # r_c' = pi_c'(a|s) / pi_old,c'(a|s), weighted by pi_old,c'(a|s) / pi_bar(a|s) (a was drawn from pi_bar, not
                # pi_c'), averaged over the trained passes.
                log_pi_all = jax.nn.log_softmax(inter['intermediates']['logits_all'][0], axis=-1)  # [B, T, A]
                lpa = jnp.take_along_axis(log_pi_all, data.action[:, None, None].astype(jnp.int32), axis=-1)[..., 0]
                T = lpa.shape[-1]
                train = (ex['halt_old_log_pc'] > -1e8) & (jnp.arange(1, T + 1) <= ex['compute_time'][:, None] + args.halt_extend)
                w = jnp.where(train, jnp.exp(ex['halt_old_log_pa'] - ex['halt_log_mu_a'][:, None]), 0.0)
                log_r = jnp.where(train, lpa - ex['halt_old_log_pa'], 0.0)
                r = jnp.exp(log_r)
                adv = advantages[:, None]
                surr = jnp.minimum(r * adv, jnp.clip(r, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * adv)
                policy_loss = -jnp.mean(jnp.sum(w * surr, axis=-1) / jnp.maximum(jnp.sum(train, axis=-1), 1))
                approx_kl = jax.lax.stop_gradient(
                    jnp.mean(jnp.sum(w * (r - 1.0 - log_r), axis=-1) / jnp.maximum(jnp.sum(train, axis=-1), 1)))
                halt_metrics['halt_extend_passes'] = jnp.mean(jnp.sum(train, axis=-1))
            # Halt head: cross-entropy to the frozen target q(c|s) built in learn_step (zero on forced-zero c).
            q = ex['halt_target']
            lpc = inter['intermediates']['halt_log_probs'][0]
            halt_tpo_loss = -jnp.mean(jnp.sum(jnp.where(q > 0, q * lpc, 0.0), axis=-1))
            policy_loss = policy_loss + halt_tpo_loss
            T = lpc.shape[-1]
            halt_metrics.update({
                'halt_tpo_loss': halt_tpo_loss,
                # KL(q || p): how far the target still is from the current p(c|s).
                'halt_tpo_kl': jax.lax.stop_gradient(halt_tpo_loss + jnp.mean(
                    jnp.sum(jnp.where(q > 0, q * jnp.log(jnp.maximum(q, 1e-30)), 0.0), axis=-1))),
                'halt_expected_c': jnp.mean(jnp.sum(jnp.exp(lpc) * jnp.arange(1, T + 1), axis=-1)),
                'halt_target_expected_c': jnp.mean(jnp.sum(q * jnp.arange(1, T + 1), axis=-1)),
                'halt_action_is_weight': jnp.mean(is_c),
            })
        else:
            policy_loss, v_loss, entropy_loss, approx_kl = ppo_terms(
                distrax.Categorical(logits=logits), values, halt_log_prob
            )
        entropy_loss = entropy_loss - args.halt_entropy_cost * halt_entropy

        think_iters = inter['intermediates']['fp_iterations'][0]

        total_loss = policy_loss + v_loss + entropy_loss

        return total_loss, {
            **halt_metrics,
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl,
            'think_iters': jnp.mean(think_iters.astype(jnp.float32)),
            # Mean predicted value vs. mean (lambda-return) target: a growing gap shows expectile optimism.
            'value_mean': jnp.mean(values),
            'value_target_mean': jnp.mean(value_targets),
        }

    use_loss_scale = args.compute_dtype == 'float16' and args.dynamic_loss_scale

    @jax.jit
    def learn_step(training_state, data, key_sgd):

        def _learn_step(carry, unused_t):

            def _train_minibatch_step(carry, data):
                training_state, key, kl_stop = carry
                key, key_loss = jax.random.split(key)

                if use_loss_scale:
                    scale = training_state.loss_scale

                    def scaled_loss(*a):
                        loss, aux = compute_ppo_loss(*a)
                        return loss * scale, aux

                    (_, metrics), grads = jax.value_and_grad(scaled_loss, has_aux=True)(
                        training_state.params, obs_normalizer(training_state), data, key_loss
                    )
                    grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
                    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
                    # Zero a nonfinite gradient so the (discarded) update below stays finite.
                    grads = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grads)
                else:
                    (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(
                        training_state.params, obs_normalizer(training_state), data, key_loss
                    )
                grad_norm = optax.global_norm(grads)  # pre-clip
                metrics['grad_norm'] = grad_norm
                if args.max_grad_norm > 0.0:
                    metrics['grad_clip_frac'] = (grad_norm > args.max_grad_norm).astype(jnp.float32)
                new_training_state = training_state.apply_gradients(grads=grads)
                if use_loss_scale:
                    # Skip the update on a nonfinite gradient and halve the scale; double it after loss_scale_period
                    # consecutive finite steps.
                    new_training_state = jax.tree_util.tree_map(
                        lambda new, old: jnp.where(finite, new, old), new_training_state, training_state
                    )
                    good = jnp.where(finite, training_state.loss_scale_good_steps + 1, 0)
                    grow = good >= args.loss_scale_period
                    new_scale = jnp.where(finite, jnp.where(grow, scale * 2.0, scale), jnp.maximum(scale / 2.0, 1.0))
                    new_training_state = new_training_state.replace(
                        loss_scale=new_scale, loss_scale_good_steps=jnp.where(grow, 0, good)
                    )
                    metrics['loss_scale'] = scale
                    metrics['nonfinite_grad_frac'] = (~finite).astype(jnp.float32)
                if args.target_kl > 0.0:
                    # Once approx_kl exceeds the target, skip every remaining
                    # minibatch update until the next rollout.
                    new_training_state = jax.tree_util.tree_map(
                        lambda new, old: jnp.where(kl_stop, old, new),
                        new_training_state,
                        training_state,
                    )
                    metrics['kl_stop_frac'] = kl_stop.astype(jnp.float32)
                    kl_stop = kl_stop | (metrics['approx_kl'] > args.target_kl)
                training_state = new_training_state

                return (training_state, key, kl_stop), metrics

            training_state, data, value_targets, advantages, key, kl_stop = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def shuffle_and_reshape(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (args.num_minibatches_per_rollout, -1) + x.shape[2:])
                return x

            batch_data = (data, value_targets, advantages)
            shuffled_batch_data = jax.tree_util.tree_map(shuffle_and_reshape, batch_data)

            (training_state, _, kl_stop), metrics = jax.lax.scan(
                _train_minibatch_step,
                (training_state, key_grad, kl_stop),
                shuffled_batch_data,
                length=args.num_minibatches_per_rollout,
            )
            return (training_state, data, value_targets, advantages, key, kl_stop), metrics

        # calculate gae
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        _, bootstrap_value = actor_critic_network.apply(
            training_state.params, terminal_obs, obs_normalizer(training_state)
        )

        rewards = data.reward * args.reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        # A decision taking c_t think iterations spends c_t - 1 extra compute steps before
        # its reward: r_t is discounted by discount_compute^(c_t - 1), and the bootstrap by
        # discount_act * discount_compute^(c_t - 1), so the cumulative discount at step t is
        # discount_act^t * discount_compute^z_t.
        compute_time = data.extras['policy_extras']['compute_time'].astype(jnp.float32)
        compute_discount = args.discount_compute ** (compute_time - 1)
        rewards = rewards * compute_discount
        discount = args.discount_act * compute_discount

        value_targets, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=data.value,
            bootstrap_value=bootstrap_value,
            lambda_=args.gae_lambda,
            discount=discount,
        )
        if halt_all_c:
            # Runtime factorization: c only scales the return of a decision (and everything after it) by
            # discount_compute^(c - 1), so Q(s, c, a) = discount_compute^(c - 1) Q_env(s, a). Recover Q_env from the
            # sampled c's advantage and evaluate every c: A_c = discount_compute^(c - 1) Q_env - V, [T_roll, B, T].
            # (GAE's multi-step return follows the behaviour think times; with a small halt_explore it is ~ on-policy.)
            q_env = (advantages + data.value) / compute_discount
            c_all = jnp.arange(1, args.max_think_iters + 1, dtype=jnp.float32)
            adv_all = (args.discount_compute ** (c_all - 1) * q_env[..., None] - data.value[..., None]) \
                * (1 - truncation)[..., None]  # as GAE: no advantage at truncated steps
            if args.normalize_advantage:
                # One shift and scale for every c (a shared constant baseline keeps the estimator unbiased).
                adv_all = (adv_all - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = adv_all
        else:
            if halt_tpo:
                # TPO target over think times. The value of halting at c is u_c = g_c^(c-1) sum_a pi_c(a|s) Q_env(s, a);
                # from the one sampled a ~ pi_bar, with V as a control variate,
                #   u_c ~= g_c^(c-1) (V + pi_old,c(a|s) / pi_bar(a|s) (Q_env - V))   (unbiased),
                # so a c whose readout would not play a gets no credit for it. q(c|s) ∝ p_old(c|s) exp(u_c / (eta * std(A))),
                # frozen for the whole update; forced-zero c stay at 0 and truncated steps keep q = p_old.
                ex = data.extras['policy_extras']
                q_env = (advantages + data.value) / compute_discount
                rho = jnp.exp(ex['halt_old_log_pa'] - ex['halt_log_mu_a'][..., None])  # [T_roll, B, T]
                c_all = jnp.arange(1, args.max_think_iters + 1, dtype=jnp.float32)
                u = args.discount_compute ** (c_all - 1) * (data.value[..., None] + rho * (q_env - data.value)[..., None])
                u = u * (1 - truncation)[..., None]
                valid = ex['halt_old_log_pc'] > -1e8  # forced-zero think times have log p = -1e9 (models/fprm_mem_halt.py)
                tgt_logits = jnp.where(
                    valid, ex['halt_old_log_pc'] + u / (args.halt_tpo_eta * (advantages.std() + 1e-8)), -jnp.inf)
                halt_target = jax.lax.stop_gradient(jax.nn.softmax(tgt_logits, axis=-1))
                data = data._replace(extras={**data.extras, 'policy_extras': {**ex, 'halt_target': halt_target}})
            if args.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        (training_state, _, _, _, _, _), metrics = jax.lax.scan(
            _learn_step,
            (training_state, data, value_targets, advantages, key_sgd,
             jnp.zeros((), bool)),
            (),
            length=args.num_epochs_per_rollout,
        )

        return training_state, metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None
    for ts in range(1, args.num_training_step + 1):

        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, training_data = data_collect_step(training_state, env_state, key_generate_unroll)
        data_collect_step_time += time.time() - data_collect_start

        learn_step_start = time.time()
        training_state, training_metrics = learn_step(training_state, training_data, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, training_metrics
            )

        if args.num_reset_steps > 0 and ts % args.num_training_steps_per_real_reset == 0:
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs)

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval

            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction': data_collect_step_time / training_step_time,
                'training/learning_time_fraction': learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                **{f'training/{name}': value for name, value in metrics.items()},
            }

            metrics = evaluator.run_evaluation(
                policy_params={'policy': training_state.params, 'normalizer': obs_normalizer(training_state)},
                training_metrics=metrics,
            )

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                import wandb
                wandb.log(metrics, step=es)
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl",
                    params=(
                        training_state.params,
                        training_state.normalizer_params,
                    ),
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
