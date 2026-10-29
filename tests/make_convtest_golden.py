"""Writes tests/data/fprm_mem_halt_golden.npz: outputs of the existing halting heads, for test_convtest_head's
bit-identity check (run once on the code before halt_head='convtest' was added, under jax 0.5.3; regenerated
under the project .venv's jax 0.10.2 after the current code reproduced the 0.5.3 file bit for bit). XLA versions
differ in the last bit, so the file records the jax version and the check skips under any other."""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_convtest_head import GOLDEN, GOLDEN_CONFIGS, golden_outputs  # noqa: E402

if __name__ == '__main__':
    jax.config.update('jax_platforms', 'cpu')
    out = {f'{i}/{k}': np.asarray(v) for i, kw in enumerate(GOLDEN_CONFIGS) for k, v in golden_outputs(kw).items()}
    out['jax_version'] = np.array(jax.__version__)
    np.savez(GOLDEN, **out)
    print(f'wrote {len(out)} arrays to {GOLDEN}')
