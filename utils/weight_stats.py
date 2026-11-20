"""Singular-value statistics of the weight matrices, for logging.

Each 2D weight matrix (both sides > 1) gives its largest and smallest singular value; across the matrices of a group
(actor = everything outside 'critic', critic = the 'critic' submodule) we log the 10th / 50th / 90th percentiles.
"""
import jax
import jax.numpy as jnp
import numpy as np


def singular_value_stats(params):
    groups = {}
    for path, w in jax.tree_util.tree_flatten_with_path(params)[0]:
        if w.ndim != 2 or min(w.shape) < 2:
            continue
        s = np.linalg.svd(np.asarray(w, np.float32), compute_uv=False)
        name = 'critic' if 'critic' in jax.tree_util.keystr(path) else 'actor'
        groups.setdefault(name, []).append((s[0], s[-1]))
    out = {}
    for name, sv in groups.items():
        sv = np.array(sv)
        for i, which in enumerate(('max', 'min')):
            for q in (10, 50, 90):
                out[f'weights/{name}_sv_{which}_p{q}'] = float(np.percentile(sv[:, i], q))
    return out
