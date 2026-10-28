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
from models.fprm_mem_halt import FPRMMemHaltActorValue, convtest_fisher
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
    slidingpuzzle_min_random_moves: int = 0       # slidingpuzzle-*: > 0 draws each training scramble length uniformly from
                                                  # [this, num_random_moves] (eval_reset stays fixed; 0 = fixed length)
    slidingpuzzle_random_moves_zipf: float = 0.0  # slidingpuzzle-*, min_random_moves > 0: p(length k) ∝ k^-this (0 = uniform)
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
    value_expectile: float = 0.5          # value loss |tau - 1(u < 0)| u^2 with u = target - V: 0.5 = MSE (mean return),
                                          # > 0.5 = an upper expectile of the return (optimistic V, as Stoix --expectile)
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    normalize_observations: bool = False  # running mean/std normalization of the observations (per input feature, so tied to the
                                          # board size); off: the models see raw observations and checkpoints store an identity normalizer
    # stabilization (all off by default)
    max_grad_norm: float = 0.0            # 0 = no gradient clipping
    target_kl: float = 0.0                # 0 = no KL early stop; else skip remaining minibatch updates in a rollout once approx_kl_action
                                          # (KL of the action policy pi_c(a|s) only, not the think time c) exceeds this
    anneal_lr: bool = False               # linearly decay lr to 0 over training

    # model
    # 'fprm'         — looped FPRM with policy-KL adaptive halting
    # 'fprm_mem'       — fprm's looped board plus a SEP token and a memory of thought tokens; always runs max_think_iters passes
    #                    and acts with the last (models/fprm_mem_halt.py with halting=False)
    # 'fprm_mem_halt'  — the same trunk with learned stochastic halting: chains of decisions k ~ p_n(k) over 0..halt_max_skip
    #                    more passes; the PPO ratio uses the joint log p(path|s) + log pi_c(a|s)
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
    fprm_expansion: float = 4.0           # MLP hidden = expansion * d_model (* 2/3 for swiglu), rounded up to mlp_round_to
    fprm_num_heads: int = 4
    fprm_num_layers: int = 1
    max_think_iters: int = 8              # cap on core calls ("think iterations") per decision
    min_think_iters: int = 2              # halting is checked only after this many iterations
    halt_criterion: str = 'kl'            # 'kl' (policy-KL halting) | 'latent_residual' (original FPRM-paper rule: latent max-token residual)
    halt_kl: float = 1e-3                 # stop thinking once KL(pi_i || pi_{i-1}) between consecutive policy readouts <= this
                                          # (0 = only at a fixed point, < 0 = never: the iso_* arms pass -1)
    halt_residual_thresh: float = 0.1     # halt_criterion='latent_residual': stop once the latent residual < this (paper fp_thresh)
    halt_head_grad: bool = False          # fprm_mem_halt: let the halting loss backpropagate into the trunk through the halting head
    halt_kl_reverse: bool = False         # fprm_mem_halt: halting KL_i = KL(pi_{i-1} || pi_i), the expectation under the acting policy
    stable_kl: bool = False               # fprm_mem_halt: halting KL from the logit difference, no fp32 rounding floor (models/stable_softmax.py)
    stable_softmax: bool = False          # fprm_mem / fprm_mem_halt: collapse-free log-softmax for the action policy
                                           # (and, fprm_mem_halt only, the halting head)
    halt_signal: str = 'kl'               # fprm_mem_halt: what the distance / raw heads compare: 'kl' (the policies) | 'latent' (the readout latents, L2)
    halt_max_skip: int = 1                # fprm_mem_halt: each halting decision picks 0..this more passes (0 = halt; 1 = yes/no at every pass)
    readout_token: str = 'thought'        # fprm_mem / fprm_mem_halt: readouts from the newest thought | 'sep' (the carried SEP token after the cells, present either way)
    value_token: str = ''                 # fprm_mem / fprm_mem_halt: '' (= readout_token) | 'thought' | 'sep': where the value is read from
    halt_token: str = ''                  # fprm_mem_halt: '' (= readout_token) | 'thought' | 'sep': the halting head's h_n (halt_head=none or halt_signal=latent)
    value_pass: str = 'last'              # fprm_mem_halt: 'last' (value from pass T) | 'halt' (from the acting pass c)
    act_iters: Optional[int] = None       # fprm_mem: act with this pass c <= max_think_iters (None = max_think_iters); with
                                          # value_pass=last the value still reads pass T (a fixed-c actor with the T-pass critic).
                                          # Logged as c for fprm_mem and None for every other architecture (wandb group-by)
    value_deep_supervision: bool = False  # fprm_mem / fprm_mem_halt: regress the value of every pass up to the value's
                                          # readout pass (T, or c with value_pass=halt) on the same target: V(s) at every depth
    halt_head: str = 'distance'           # fprm_mem_halt: halting MLP on h_n ('none'), on log S_(n+1) ('distance') or on [z_n, z_(n+1)] ('raw');
                                          # 'convtest': learned convergence test theta0 + theta1 f_n (models/fprm_mem_halt.py)
    halt_offset: str = 'none'             # convtest: 'uniform' adds o_n = -log(T - n) to the halt logit (c ~ Uniform at init)
    halt_replay_features: bool = True     # convtest: the loss replays the rollout's features f instead of recomputing them
    halt_precondition: bool = True        # convtest: update (theta0, theta1) by SGD on Sigma^-1 g (Fisher of the decisions);
                                          # off: Adam with halt_lr
    halt_lr: float = 1e-2                 # convtest: learning rate of (theta0, theta1)
    halt_damping: float = 1e-3            # convtest: damping added to the Fisher Sigma
    halt_head_width: int = 32             # fprm_mem_halt: hidden width of the halting MLP
    halt_entropy_cost: float = 0.0        # fprm_mem_halt: entropy bonus on p(c|s) (the action entropy keeps entropy_cost)
    halt_objective: str = 'pg'            # fprm_mem_halt: 'pg' = log p(path) inside PPO's ratio; 'next' = the softmax over k
                                          # at the sampled c against the passes it leads to, J_(c+k) - J_c (see the loss)
    halt_clip: str = 'joint'              # halt_objective='pg': PPO clip of one ratio over log p(path) + log pi_c(a|s) ('joint'),
                                          # or of pi_c(a|s) apart from p(path) as a whole ('path') or from each decision ('each')
    halt_rho_max: float = 2.0             # halt_objective='next': clip of the ratio pi_n(a|s) / mu(a|s) in J_n
    halt_prior_coef: float = 0.0          # fprm_mem_halt: weight of KL(p_n || q) at every decision pass, decayed linearly
                                          # to 0 over training; q favours continuing (0 = off)
    halt_prior_halt: float = 0.1          # q(k = 0); q(k) = (1 - q(0)) / K for every k >= 1
    cot_thought_norm: bool = False        # fprm_mem: RMS-normalize each thought before feeding it back
    init_checkpoint: str = ""             # if set, initializes every param group the checkpoint has (e.g. an iso_mem
                                           # checkpoint's core/actor_head/value_head) instead of a fresh random init;
                                           # groups the checkpoint lacks (e.g. fprm_mem_halt's halt_hidden/halt_out
                                           # on an iso_mem checkpoint) stay freshly initialized. Also loads its
                                           # normalizer_params. Expects the (params, normalizer_params) tuple format
                                           # save_params()/load_params() use.
    halt_adam_b1: float = 0.9             # fprm_mem_halt: Adam beta_1 / beta_2 of the halting head (halt_hidden/halt_out) only;
    halt_adam_b2: float = 0.999           # the rest keeps optax's 0.9 / 0.999 (arXiv 2601.21739 suggests 0.99 / 0.99)
    freeze_except_halt: bool = False      # fprm_mem_halt: zero the optimizer update for every param group except
                                           # halt_hidden/halt_out - trains only the halting head, typically paired
                                           # with init_checkpoint pointed at a halting=False checkpoint


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


def value_loss(values, targets, expectile, mask=None):
    """Expectile regression (tau = 0.5 gives the MSE 0.25 * mean(v_error^2)). values [T, B] with mask [T, B]: each
    state's loss is the mean over its masked passes."""
    v_error = targets - values
    loss = jnp.abs(expectile - (v_error < 0).astype(v_error.dtype)) * v_error * v_error * 0.5
    if mask is None:
        return jnp.mean(loss)
    return jnp.mean(jnp.sum(jnp.where(mask, loss, 0.0), axis=0) / jnp.sum(mask, axis=0))


def count_parameters(params):
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))


def save_params(path, params):
    with open(path, 'wb') as fout:
        fout.write(pickle.dumps(params))


def load_params(path):
    with open(path, 'rb') as fin:
        return pickle.loads(fin.read())


def make_inference_fn(actor_critic_network, stochastic_halt: bool = False):
    """Creates params and inference function for the PPO agent.

    stochastic_halt: the network samples its think time c (fprm_mem_halt); the stored log_prob is the joint
    log p(c|s) + log pi_c(a|s), and compute_time = c is replayed in the loss. Deterministic: c = argmax p(c|s).
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
            action_log_prob = policy_dist.log_prob(actions)
            log_prob = action_log_prob
            if stochastic_halt:
                log_prob = log_prob + inter['intermediates']['halt_log_prob'][0]
            extras = {
                'log_prob': log_prob,
                'action_log_prob': action_log_prob,  # log pi_c(a|s) alone (= log_prob without stochastic halting)
                'value': value,
                'compute_time': compute_time,
            }
            if 'halt_path' in inter['intermediates']:  # fprm_mem_halt: the loss replays the decision passes too
                extras['halt_path'] = inter['intermediates']['halt_path'][0]
                extras['halt_decision_log_probs'] = inter['intermediates']['halt_decision_log_probs'][0]
            if 'halt_feature_t' in inter['intermediates']:  # convtest: the features the decisions read, [B, T]
                extras['halt_features'] = inter['intermediates']['halt_feature_t'][0].T
            return actions, extras

        return policy
    return make_policy


HALT_PARAMS = ('halt_hidden', 'halt_out', 'theta0', 'theta1')  # the halting head's param groups


def make_optimizer(args, init_params=None):
    """init_params is only needed (and required) when args.freeze_except_halt or the halting head's Adam betas are not
    the defaults - its pytree structure labels halt_hidden/halt_out apart from every other param group."""
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
    clip = lambda tx: optax.chain(optax.clip_by_global_norm(args.max_grad_norm), tx) if args.max_grad_norm > 0.0 else tx
    halt_tx = optax.adam(learning_rate=learning_rate, b1=args.halt_adam_b1, b2=args.halt_adam_b2)
    halt_betas = (args.halt_adam_b1, args.halt_adam_b2) != (0.9, 0.999)

    def labels(other):  # 'train' for the halting head's param groups, `other` for the rest
        return {'params': {k: jax.tree_util.tree_map(lambda _: 'train' if k in HALT_PARAMS else other, v)
                           for k, v in init_params['params'].items()}}

    if args.halt_head == 'convtest' and not args.freeze_except_halt:
        # (theta0, theta1): SGD on the preconditioned gradient Sigma^-1 g (set in learn_step), or Adam; clipping covers
        # the rest only, so the preconditioned head gradient does not scale the trunk's update.
        head = optax.sgd(args.halt_lr) if args.halt_precondition else optax.adam(args.halt_lr, b1=args.halt_adam_b1, b2=args.halt_adam_b2)
        return optax.multi_transform({'train': head, 'rest': clip(optax.adam(learning_rate=learning_rate))}, labels('rest'))

    if args.freeze_except_halt:
        assert init_params is not None, "freeze_except_halt needs init_params to build the freeze mask"
        # optax.masked() passes ungated gradients through unchanged for mask=False leaves - not a freeze. Use
        # multi_transform instead: 'frozen' groups get optax.set_to_zero(), an actual zero update.
        return optax.multi_transform({'train': clip(halt_tx), 'frozen': optax.set_to_zero()}, labels('frozen'))
    if halt_betas:  # the halting head gets its own Adam moments; clipping stays global over all params
        assert init_params is not None, "halt_adam_b1 / halt_adam_b2 need init_params to label the halting head"
        return clip(optax.multi_transform({'train': halt_tx, 'rest': optax.adam(learning_rate=learning_rate)}, labels('rest')))
    return clip(optax.adam(learning_rate=learning_rate))


def main(args: Args):

    assert 0.0 < args.value_expectile < 1.0, args.value_expectile
    args.num_training_step = args.num_timesteps // (args.num_envs * args.rollout_length)
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    args.minibatch_size = args.num_envs * args.rollout_length // (args.num_minibatches_per_rollout)

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")

    if args.architecture == 'fprm_mem':
        args.act_iters = args.act_iters or args.max_think_iters
    else:
        assert args.act_iters is None, 'act_iters is only for fprm_mem'

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
        expansion=args.fprm_expansion,
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
    elif args.architecture in ('fprm_mem', 'fprm_mem_halt'):
        assert args.halt_criterion == 'kl', "fprm_mem / fprm_mem_halt support only halt_criterion='kl'"
        assert args.halt_clip in ('joint', 'path', 'each'), f"halt_clip must be joint, path or each, got {args.halt_clip}"
        actor_critic_network = FPRMMemHaltActorValue(
            halting=args.architecture == 'fprm_mem_halt',
            m=env.m,
            n=env.n,
            config=fprm_config,
            max_think_iters=args.max_think_iters,
            min_think_iters=args.min_think_iters,
            halt_head_grad=args.halt_head_grad,
            halt_kl_reverse=args.halt_kl_reverse,
            stable_kl=args.stable_kl,
            stable_softmax=args.stable_softmax,
            halt_signal=args.halt_signal,
            halt_head=args.halt_head,
            halt_offset=args.halt_offset,
            halt_head_width=args.halt_head_width,
            readout_token=args.readout_token,
            value_token=args.value_token,
            halt_token=args.halt_token,
            value_pass=args.value_pass,
            act_iters=args.act_iters or 0,
            halt_max_skip=args.halt_max_skip,
            output_dim_1=action_size,
            obs_channels=obs_channels,
            action_head=action_head,
            tokenizer=args.tokenizer,
            tokenizer_conv_features=args.tokenizer_conv_features,
            thought_norm=args.cot_thought_norm,
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

    init_params = actor_critic_network.init(key_policy, x=jnp.zeros((1, obs_size + goal_size)))
    init_normalizer = running_statistics.init_state((obs_size + goal_size,))
    if args.init_checkpoint:
        ckpt_params, ckpt_normalizer = load_params(args.init_checkpoint)
        # Overwrite every param group the checkpoint has; a group it lacks (e.g. fprm_mem_halt's halt_hidden/
        # halt_out on a halting=False checkpoint) stays at its fresh init above.
        merged = dict(init_params['params'])
        merged.update(ckpt_params['params'])
        init_params = {'params': merged}
        init_normalizer = ckpt_normalizer

    training_state = PPOTrainingState.create(
        apply_fn=None,
        params=init_params,
        tx=make_optimizer(args, init_params),
        normalizer_params=init_normalizer,
        env_steps=np.zeros((), dtype=np.float64),
        loss_scale=jnp.asarray(args.loss_scale_init, jnp.float32),
        loss_scale_good_steps=jnp.zeros((), jnp.int32),
    )
    stochastic_halt = args.architecture == 'fprm_mem_halt'
    make_policy = make_inference_fn(actor_critic_network, stochastic_halt)

    print(f'\nNumber of parameters in actor critic network are: {count_parameters(training_state.params)}\n')

    # Evaluators: greedy (eval/: most likely action and think time) and sampled (eval_sample/: the rollout policy)
    eval_kw = dict(num_eval_envs=args.num_eval_envs, episode_length=episode_length, test_eval_env=test_eval_env,
                   test_episode_length=test_episode_length, discount_act=args.discount_act,
                   discount_compute=args.discount_compute)
    evaluator = Evaluator(eval_env, functools.partial(make_policy, deterministic=True), key=key_eval, **eval_kw)
    sample_evaluator = Evaluator(eval_env, make_policy, key=jax.random.fold_in(key_eval, 1), prefix='eval_sample',
                                 **eval_kw)

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

    def compute_ppo_loss(params, normalizer_params, data, rng, progress):
        data, value_targets, advantages = data

        next_halt = stochastic_halt and args.halt_objective == 'next'
        # The rollout's log mu(c, a | s), or log mu(a | s) alone when the halting head has its own objective.
        behaviour_action_log_probs = data.extras['policy_extras']['action_log_prob' if next_halt else 'log_prob']

        halt_clip = args.halt_clip if stochastic_halt and not next_halt else 'joint'

        def surrogate(log_rho):  # PPO's clipped surrogate of one ratio
            rho = jnp.exp(log_rho)
            return jnp.minimum(rho * advantages, jnp.clip(rho, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)

        def ppo_terms(policy_dist, baseline, halt_log_prob=0.0, halt_decision_log_probs=None):
            # Policy function loss (stochastic halting: joint log p(c|s) + log pi_c(a|s))
            target_pi_log_probs = policy_dist.log_prob(data.action)
            target_action_log_probs = target_pi_log_probs + halt_log_prob
            if halt_clip == 'joint':
                policy_loss = -jnp.mean(surrogate(target_action_log_probs - behaviour_action_log_probs))
            else:  # the action's surrogate, plus the halting path's ('path') or each decision's ('each')
                mu_pi = data.extras['policy_extras']['action_log_prob']
                policy_loss = -jnp.mean(surrogate(target_pi_log_probs - mu_pi))
                if halt_clip == 'path':
                    policy_loss -= jnp.mean(surrogate(halt_log_prob - (behaviour_action_log_probs - mu_pi)))
                else:
                    mu_dec = data.extras['policy_extras']['halt_decision_log_probs']  # [B, T]
                    dec = (data.extras['policy_extras']['halt_path'][:, None] >> jnp.arange(mu_dec.shape[-1])) & 1
                    s_dec = jax.vmap(surrogate, in_axes=1, out_axes=1)(halt_decision_log_probs - mu_dec)
                    policy_loss -= jnp.mean(jnp.sum(jnp.where(dec == 1, s_dec, 0.0), axis=-1))

            v_loss = value_loss(baseline, value_targets, args.value_expectile)

            # Entropy loss
            entropy = jnp.mean(policy_dist.entropy())
            entropy_loss = args.entropy_cost * -entropy

            # k3 estimator of KL(behaviour || target), Schulman (2020): of the joint (c, a) policy, and of the action
            # policy pi_c(a|s) alone (at the replayed c; the target_kl early stop uses this one).
            def k3(log_rho):
                return jax.lax.stop_gradient(jnp.mean(jnp.exp(log_rho) - 1.0 - log_rho))
            approx_kl = k3(target_action_log_probs - behaviour_action_log_probs)
            approx_kl_action = k3(target_pi_log_probs - data.extras['policy_extras']['action_log_prob'])
            return policy_loss, v_loss, entropy_loss, approx_kl, approx_kl_action

        # PPO loss on the final actor / value readout. Halted examples freeze
        # inside the network, so their readout is the one they halted at.
        # Stochastic halting: replay the rollout's think time c.
        halt_kw = {'halt_steps': data.extras['policy_extras']['compute_time']} if stochastic_halt else {}
        if 'halt_path' in data.extras['policy_extras']:
            halt_kw['halt_path'] = data.extras['policy_extras']['halt_path']
        if args.halt_replay_features and 'halt_features' in data.extras['policy_extras']:
            halt_kw['halt_features'] = data.extras['policy_extras']['halt_features']
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
            halt_metrics = {'halt_entropy': halt_entropy}
            # mean p(k) over passes and states, and the number of decisions per state
            p_k = jnp.mean(inter['intermediates']['halt_step_probs'][0], axis=(0, 1))
            halt_metrics.update({f'halt_step_prob/k{k}': p_k[k] for k in range(p_k.shape[0])})
            path = data.extras['policy_extras']['halt_path']
            halt_metrics['halt_decisions'] = jnp.mean(jnp.sum((path[..., None] >> jnp.arange(32)) & 1, axis=-1).astype(jnp.float32))
            # Per-pass mean p_i(k = 0), the halting probability at a decision at pass i, in its own wandb section.
            mean_h = jnp.mean(inter['intermediates']['halt_prob'][0], axis=-1)
            halt_metrics.update({f'halt_prob/pass_{i + 1:02d}': mean_h[i] for i in range(mean_h.shape[0])})
            # Latent residual between consecutive passes at the sampled (halted) c.
            c_idx = (data.extras['policy_extras']['compute_time'] - 1).astype(jnp.int32)
            res_c = jnp.take_along_axis(inter['intermediates']['latent_residual_t'][0], c_idx[None], axis=0)
            halt_metrics['latent_res_at_c'] = jnp.mean(res_c)
            if args.halt_head == 'convtest':
                theta0, theta1 = params['params']['theta0'], params['params']['theta1']
                safe = jnp.abs(theta1) > 1e-6
                halt_metrics.update({'halt_theta0': theta0, 'halt_theta1': theta1, 'halt_kappa': -theta1,
                                     'halt_log_eps': jnp.where(safe, -theta0 / jnp.where(safe, theta1, 1.0), jnp.nan)})
                # Fisher of the decisions on the sampled paths (passes n <= T - 1), for the preconditioned head update
                f = inter['intermediates']['halt_feature_t'][0][:-1]  # [T - 1, B]
                p = jnp.exp(inter['intermediates']['halt_step_log_probs'][0][:-1, :, 0])
                dec = ((path[None] >> jnp.arange(f.shape[0])[:, None]) & 1) == 1
                halt_metrics['halt_fisher'] = jax.lax.stop_gradient(convtest_fisher(f, p, dec, args.halt_damping))
        else:
            halt_log_prob, halt_entropy = 0.0, 0.0
        halt_loss = 0.0
        if next_halt:
            # The halting head's softmax over k at the rollout's think time c, scored by where each k leads: J_n =
            # discount_compute^(n - c) * min(pi_n(a|s) / mu(a|s), rho_max) * G estimates the return of acting with pass n
            # (mu the behaviour policy, G the return target, which carries discount_compute^(c - 1)), each k is scored as a
            # halt at c + k (the cap: T), and the head maximizes sum_k p_c(k) (J_(c+k) - J_c). A think time at the cap
            # (c = T) has no halt decision. PPO's ratio then covers pi_c(a|s) alone.
            lp_a = jnp.take_along_axis(inter['intermediates']['policy_log_probs_t'][0], data.action[None, :, None].astype(jnp.int32), axis=-1)[..., 0]
            rho = jnp.minimum(jnp.exp(lp_a - data.extras['policy_extras']['action_log_prob']), args.halt_rho_max)  # [T, B]
            T_max = rho.shape[0]
            c = data.extras['policy_extras']['compute_time'].astype(jnp.int32)
            J = jax.lax.stop_gradient(args.discount_compute ** (jnp.arange(1, T_max + 1)[:, None] - c) * rho * value_targets)
            lp_c = jnp.take_along_axis(inter['intermediates']['halt_step_log_probs'][0], (c - 1)[None, :, None], axis=0)[0]  # [B, K + 1]
            gain = jnp.take_along_axis(J, jnp.minimum(c[None] + jnp.arange(lp_c.shape[-1])[:, None], T_max) - 1, axis=0).T  # [B, K + 1]
            gain = gain - gain[:, :1]  # J_(c+k) - J_c
            valid = c < T_max
            n_valid = jnp.maximum(jnp.sum(valid), 1)
            halt_loss = -jnp.sum(jnp.where(valid, jnp.sum(jnp.exp(lp_c) * gain, axis=-1), 0.0)) / n_valid
            best_k = jnp.argmax(gain, axis=-1)
            halt_metrics.update({'halt_loss': halt_loss, 'halt_rho_clip_frac': jnp.mean(rho >= args.halt_rho_max),
                                 'halt_best_k': jnp.sum(jnp.where(valid, best_k, 0)) / n_valid,
                                 'halt_continue_better_frac': jnp.sum(valid & (best_k > 0)) / n_valid})
            halt_log_prob = 0.0
        if stochastic_halt and args.halt_prior_coef > 0:
            # Prior on the halting head favouring "continue": KL(p_n || q) at every pass that can decide
            # (min_think_iters .. T - 1), q(0) = halt_prior_halt and q(k) = (1 - q(0)) / K for k >= 1, weighted by
            # halt_prior_coef * (1 - progress), so it fades out by the end of training.
            lp = inter['intermediates']['halt_step_log_probs'][0][args.min_think_iters - 1:-1]  # [decision passes, B, K + 1]
            K = lp.shape[-1] - 1
            log_q = jnp.log(jnp.array([args.halt_prior_halt] + [(1 - args.halt_prior_halt) / K] * K))
            prior_kl = jnp.mean(jnp.sum(jnp.exp(lp) * (lp - log_q), axis=-1))
            prior_coef = args.halt_prior_coef * jnp.maximum(1.0 - progress, 0.0)
            halt_loss = halt_loss + prior_coef * prior_kl
            halt_metrics.update({'halt_prior_kl': prior_kl, 'halt_prior_coef': prior_coef})
        if 'policy_kl_t' in inter['intermediates']:  # fprm_mem / fprm_mem_halt
            # Per-pass means, each in its own wandb section: log10 of the policy KL_i between passes i and i-1 (KL spans
            # decades, so the mean of the log) and the latent residual.
            log_kl = jnp.mean(jnp.log10(jnp.maximum(inter['intermediates']['policy_kl_t'][0], 0.0) + 1e-10), axis=-1)
            halt_metrics.update({f'halt_log10_kl/pass_{i + 1:02d}': log_kl[i] for i in range(log_kl.shape[0])})
            mean_res = jnp.mean(inter['intermediates']['latent_residual_t'][0], axis=-1)
            halt_metrics.update({f'latent_res/pass_{i + 1:02d}': mean_res[i] for i in range(mean_res.shape[0])})
        policy_loss, v_loss, entropy_loss, approx_kl, approx_kl_action = ppo_terms(
            distrax.Categorical(logits=logits), values, halt_log_prob,
            inter['intermediates']['halt_decision_log_probs'][0] if halt_clip == 'each' else None,
        )
        entropy_loss = entropy_loss - args.halt_entropy_cost * halt_entropy
        if 'value_t' in inter['intermediates']:  # fprm_mem / fprm_mem_halt
            value_t = inter['intermediates']['value_t'][0]  # [T, B]
            if args.value_deep_supervision:  # passes 1..n_max all regress on the same target
                n_max = inter['intermediates']['fp_iterations'][0] if args.value_pass == 'halt' else value_t.shape[0]
                v_loss = value_loss(value_t, value_targets, args.value_expectile, jnp.arange(1, value_t.shape[0] + 1)[:, None] <= n_max)
            # Per-pass value loss over every state (supervised or not), each in its own wandb section.
            v_loss_t = jax.lax.stop_gradient(jax.vmap(lambda v: value_loss(v, value_targets, args.value_expectile))(value_t))
            halt_metrics.update({f'v_loss/pass_{i + 1:02d}': v_loss_t[i] for i in range(v_loss_t.shape[0])})

        think_iters = inter['intermediates']['fp_iterations'][0]

        total_loss = policy_loss + v_loss + entropy_loss + halt_loss

        return total_loss, {
            **halt_metrics,
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl,
            'approx_kl_action': approx_kl_action,
            'think_iters': jnp.mean(think_iters.astype(jnp.float32)),
            # Mean predicted value vs. mean (lambda-return) target: a growing gap shows expectile optimism.
            'value_mean': jnp.mean(values),
            'value_target_mean': jnp.mean(value_targets),
        }

    use_loss_scale = args.compute_dtype == 'float16' and args.dynamic_loss_scale

    def progress(training_state):  # fraction of training done, in [0, 1]
        return (training_state.env_steps / args.num_timesteps).astype(jnp.float32)

    @jax.jit
    def learn_step(training_state, data, key_sgd):

        def _train_minibatch_step(training_state, key_loss, data):
            if use_loss_scale:
                scale = training_state.loss_scale

                def scaled_loss(*a):
                    loss, aux = compute_ppo_loss(*a)
                    return loss * scale, aux

                (_, metrics), grads = jax.value_and_grad(scaled_loss, has_aux=True)(
                    training_state.params, obs_normalizer(training_state), data, key_loss, progress(training_state)
                )
                grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
                finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
                # Zero a nonfinite gradient so the (discarded) update below stays finite.
                grads = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grads)
            else:
                (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(
                    training_state.params, obs_normalizer(training_state), data, key_loss, progress(training_state)
                )
            grad_norm = optax.global_norm(grads)  # pre-clip
            metrics['grad_norm'] = grad_norm
            if args.max_grad_norm > 0.0:
                metrics['grad_clip_frac'] = (grad_norm > args.max_grad_norm).astype(jnp.float32)
            fisher = metrics.pop('halt_fisher', None)
            if fisher is not None and args.halt_precondition:  # convtest: head gradient g -> Sigma^-1 g (SGD in the optimizer)
                p = grads['params']
                g = jnp.linalg.solve(fisher, jnp.stack([p['theta0'], p['theta1']]))
                grads = {**grads, 'params': {**p, 'theta0': g[0], 'theta1': g[1]}}
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
            return new_training_state, metrics

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
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Epochs of shuffled minibatch updates. With target_kl > 0, the first update whose approx_kl_action exceeds the target is
        # the last one of this rollout: both loops exit, so the skipped updates are never computed. Metrics are the mean
        # over the updates that ran; kl_stop_frac is the fraction skipped.
        batch_data = (data, value_targets, advantages)
        num_updates = args.num_epochs_per_rollout * args.num_minibatches_per_rollout
        minibatch = jax.tree_util.tree_map(
            lambda x: x.reshape((args.num_minibatches_per_rollout, -1) + x.shape[2:])[0], batch_data)
        metric_shapes = jax.eval_shape(_train_minibatch_step, training_state, key_sgd, minibatch)[1]
        zero_metrics = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), metric_shapes)

        def _epoch(carry):
            training_state, key, kl_stop, epoch, n, metric_sum = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
            shuffled_batch_data = jax.tree_util.tree_map(
                lambda x: jax.random.permutation(key_perm, x).reshape(
                    (args.num_minibatches_per_rollout, -1) + x.shape[2:]),
                batch_data)

            def _minibatch(carry):
                training_state, key, kl_stop, i, n, metric_sum = carry
                key, key_loss = jax.random.split(key)
                training_state, metrics = _train_minibatch_step(
                    training_state, key_loss, jax.tree_util.tree_map(lambda x: x[i], shuffled_batch_data))
                if args.target_kl > 0.0:
                    kl_stop = metrics['approx_kl_action'] > args.target_kl
                metric_sum = jax.tree_util.tree_map(jnp.add, metric_sum, metrics)
                return training_state, key, kl_stop, i + 1, n + 1, metric_sum

            training_state, _, kl_stop, _, n, metric_sum = jax.lax.while_loop(
                lambda c: (c[3] < args.num_minibatches_per_rollout) & ~c[2], _minibatch,
                (training_state, key_grad, kl_stop, 0, n, metric_sum))
            return training_state, key, kl_stop, epoch + 1, n, metric_sum

        training_state, _, _, _, n, metric_sum = jax.lax.while_loop(
            lambda c: (c[3] < args.num_epochs_per_rollout) & ~c[2], _epoch,
            (training_state, key_sgd, jnp.zeros((), bool), 0, 0, zero_metrics))
        metrics = jax.tree_util.tree_map(lambda s: s / n, metric_sum)
        if args.target_kl > 0.0:
            metrics['kl_stop_frac'] = 1.0 - n / num_updates

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
                **{name if '/' in name else f'training/{name}': value for name, value in metrics.items()},  # 'a/b': own section
            }

            eval_params = {'policy': training_state.params, 'normalizer': obs_normalizer(training_state)}
            metrics = evaluator.run_evaluation(policy_params=eval_params, training_metrics=metrics)
            metrics = sample_evaluator.run_evaluation(policy_params=eval_params, training_metrics=metrics)

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
