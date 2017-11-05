"""
Some tests for model related functions/modules
"""

import numpy as np
import src.models as models


def test_dem_basic():
    """
    Basic sanity check for dem
    """

    N = 100 # 100 observations
    M = 15  # 15 models

    scores = np.random.rand(N)
    all_scores = np.repeat(scores[:, None], M, 1)

    weights = models.dem(all_scores)
    expected_weights = np.ones(M) / M
    assert np.allclose(weights, expected_weights)
