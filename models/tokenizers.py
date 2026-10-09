"""Alternative observation tokenizers, for ablating the per-cell tokenization.

Every model (models/fprm_thinker.py, models/transformer_baseline.py,
models/fprm_cot_thinker.py) takes a `tokenizer` option:

  'cell' (default): one token per grid cell from its channels + geometry
      (`fprm_thinker.tokenize` + the model's own `cell_projection`; unchanged,
      so existing checkpoints keep their parameter tree).
  'flat': one token per observation dimension: token_i = Dense(1 -> d)(x_i) +
      index_embed[i]. No spatial structure is given beyond the learned index
      embedding.
  'conv': one 3x3 SAME convolution (ReLU) over the (m, n, obs_channels) grid,
      then one token per output channel: token_f = Dense(m*n -> d)(feature map
      f, flattened) + channel_embed[f].

The core's grid convolution assumes one token per cell, so the 'flat' and
'conv' tokenizers turn it off (`core_config`) and present their tokens to the
core as a (1, num_tokens) "grid" (`token_grid_shape`). The per-cell action
head needs per-cell tokens, so it requires 'cell'.
"""

import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from models.fprm import FPRMConfig, pytorch_depthwise_conv_init

TOKENIZERS = ('cell', 'flat', 'conv')


def check_tokenizer(tokenizer, action_head, obs_channels):
    assert tokenizer in TOKENIZERS, f'tokenizer must be one of {TOKENIZERS}, got {tokenizer!r}'
    if action_head == 'per_cell':
        assert tokenizer == 'cell', "the 'per_cell' action head needs tokenizer='cell'"
    if tokenizer == 'conv':
        assert obs_channels > 0, "tokenizer='conv' needs a cell-major plane observation (obs_channels > 0)"


def core_config(config: FPRMConfig, tokenizer) -> FPRMConfig:
    """The core's config: no grid convolution unless the tokens are grid cells."""
    return config if tokenizer == 'cell' else dataclasses.replace(config, use_grid_conv=False)


def num_tokens(tokenizer, m, n, obs_dim, conv_features):
    return {'cell': m * n, 'flat': obs_dim, 'conv': conv_features}[tokenizer]


def token_grid_shape(tokenizer, m, n, obs_dim, conv_features):
    """(R, C) the core sees the input tokens as (R * C = number of tokens)."""
    if tokenizer == 'cell':
        return (m, n)
    return (1, num_tokens(tokenizer, m, n, obs_dim, conv_features))


class InputGridConv(nn.Module):
    """Residual depthwise 3x3 convolution over [B, m*n, d] cell tokens (row-major): cells + DWConv(cells).

    The FPRM core's grid convolution acts on the latent z, which carries the board only from the second core call
    on, so models that pass the cells through the core at most once (fprm_cot, fprm_perceiver) get no
    data-dependent local mixing from it. This applies the same kind of convolution to the cell embeddings
    themselves, once per decision, so every cell token sees its 3x3 neighbourhood."""

    m: int
    n: int

    @nn.compact
    def __call__(self, cells):
        B, L, H = cells.shape
        assert L == self.m * self.n, (L, self.m, self.n)
        conv = nn.Conv(H, (3, 3), padding='SAME', feature_group_count=H, use_bias=False,
                       kernel_init=pytorch_depthwise_conv_init, name='conv')
        return cells + conv(cells.reshape(B, self.m, self.n, H)).reshape(B, L, H)


class ObsTokenizer(nn.Module):
    """'flat' / 'conv' tokenizer: [B, obs_dim] -> [B, num_tokens, d_model]."""

    tokenizer: str
    m: int
    n: int
    obs_channels: int
    d_model: int
    conv_features: int = 64

    @nn.compact
    def __call__(self, x_flat):
        B, D = x_flat.shape
        if self.tokenizer == 'flat':
            index_embed = self.param('index_embed', nn.initializers.normal(stddev=0.02), (D, self.d_model))
            return nn.Dense(self.d_model, name='value_proj')(x_flat[..., None]) + index_embed[None]
        if self.tokenizer == 'conv':
            m, n, C = self.m, self.n, self.obs_channels
            assert D == m * n * C, (D, m, n, C)
            grid = x_flat.reshape(B, m, n, C)
            features = nn.relu(nn.Conv(self.conv_features, (3, 3), padding='SAME', name='conv')(grid))
            maps = features.reshape(B, m * n, self.conv_features).transpose(0, 2, 1)  # [B, F, m*n]
            channel_embed = self.param(
                'channel_embed', nn.initializers.normal(stddev=0.02), (self.conv_features, self.d_model)
            )
            return nn.Dense(self.d_model, name='channel_proj')(maps) + channel_embed[None]
        raise ValueError(f'ObsTokenizer handles only flat / conv, got {self.tokenizer!r}')
