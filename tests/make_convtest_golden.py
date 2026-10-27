"""Writes tests/data/fprm_mem_halt_golden.npz: outputs of the existing halting heads, for test_convtest_head's
bit-identity check (run once on the code before halt_head='convtest' was added)."""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_convtest_head import GOLDEN, GOLDEN_CONFIGS, golden_outputs  # noqa: E402

if __name__ == '__main__':
    jax.config.update('jax_platforms', 'cpu')
    out = {f'{i}/{k}': np.asarray(v) for i, kw in enumerate(GOLDEN_CONFIGS) for k, v in golden_outputs(kw).items()}
    np.savez(GOLDEN, **out)
    print(f'wrote {len(out)} arrays to {GOLDEN}')
