"""
Some tests for model related functions/modules
"""

import numpy as np
import src.models as models


def test_dem_basic():
    """
    Basic sanity check for dem
    """

    N = 1000 # observations
    M = 15   # models

    scores = np.random.rand(N)
    all_scores = np.repeat(scores[:, None], M, 1)

    weights = models.dem(all_scores)
    expected_weights = np.ones(M) / M
    assert np.allclose(weights, expected_weights)


def test_dem_random_init():
    """
    Test dem with random initial weights
    """

    N = 1000 # observations
    M = 15   # models

    scores = np.random.rand(N)
    all_scores = np.repeat(scores[:, None], M, 1)

    init_weights = np.random.rand(M)
    init_weights /= np.sum(init_weights)
    weights = models.dem(all_scores, weights=init_weights)
    expected_weights = init_weights # Since all models are similar

    assert np.allclose(weights, expected_weights)


def test_dem_dominant_model():
    """
    Test dem with random initial weights and a dominant model
    """

    N = 1000 # observations
    M = 15   # models

    scores = np.zeros((N, M)) + 0.001
    scores[:, 0] = 0.999

    init_weights = np.random.rand(M)
    init_weights /= np.sum(init_weights)
    weights = models.dem(scores, weights=init_weights)
    expected_weights = np.zeros(M)
    expected_weights[0] = 1

    assert np.allclose(weights, expected_weights)


def test_dem_multiple_dominant_models():
    """
    Test dem with multiple dominant models
    """

    N = 1000 # observations
    M = 15   # models

    scores = np.zeros((N, M)) + 0.001
    scores[:, :5] = 0.999

    weights = models.dem(scores)
    expected_weights = np.zeros(M)
    expected_weights[:5] = 0.2

    assert np.allclose(weights, expected_weights)
