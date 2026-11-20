import numpy as np

from utils.weight_stats import singular_value_stats


def test_singular_value_stats():
    # diagonal matrices: singular values are the diagonal entries
    params = {'params': {'a': {'kernel': np.diag([3.0, 1.0])}, 'b': {'kernel': np.diag([5.0, 2.0, 0.5])},
                         'bias': np.ones(4), 'head': {'kernel': np.ones((8, 1))},
                         'critic': {'k': {'kernel': np.diag([7.0, 4.0])}}}}
    m = singular_value_stats(params)
    assert np.isclose(m['weights/actor_sv_max_p50'], np.percentile([3.0, 5.0], 50))
    assert np.isclose(m['weights/actor_sv_min_p10'], np.percentile([1.0, 0.5], 10))
    assert np.isclose(m['weights/critic_sv_max_p90'], 7.0) and np.isclose(m['weights/critic_sv_min_p50'], 4.0)
    assert len(m) == 12
