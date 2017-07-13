"""
Utilities for working with distributions and similar stuff
"""

import keras.backend as K
import numpy as np
import losses
from scipy.stats import norm


def dist_mean(dist, bins=np.linspace(0, 12.9, 130)):
    """
    Return the mean of distribution using default wili bins
    (skipping bin 13-100)
    """

    return np.sum(bins * dist)

def dist_max(dist, bins=np.linspace(0, 12.9, 130)):
    """
    Return value for max bin
    """

    return bins[np.argmax(dist)]

def dist_std(dist, bins=np.linspace(0, 12.9, 130)):
    """
    Return standard deviation
    """

    mean = dist_mean(dist, bins)
    var = np.sum(((bins - mean) ** 2) * dist)
    return np.sqrt(var)

def dist_median(dist, bins=np.linspace(0, 12.9, 130)):
    return np.max((np.cumsum(dist) < 0.5) * bins)

def dist_quartiles(dist, bins=np.linspace(0, 12.9, 130)):
    """
    Return quartiles division points
    """

    return np.array([
        np.max((np.cumsum(dist) < i) * bins)
        for i in [0.25, 0.5, 0.75]
    ])

def mdn_params_to_dists(params, bins=np.linspace(0, 12.9, 130)):
    """
    Convert parameters from mdn to distributions
    """

    n_mix = params.shape[1] // 3
    mu, sigma, w = losses.separate_mdn_params(params, n_mix)
    sigma = sigma.eval()
    w = w.eval()

    dists = np.zeros((params.shape[0], bins.shape[0]))

    for i in range(params.shape[0]):
        for nm in range(n_mix):
            dists[i, :] += w[i, nm] * norm(mu[i, nm], sigma[i, nm]).pdf(bins)

    dists /= dists.sum(axis=1, keepdims=True)
    return dists

def wili_to_dists(wili, bins=np.linspace(0, 12.9, 130)):
    """
    Wili values to one hot encoded bins
    """

    y = np.zeros((wili.shape[0], bins.shape[0]))

    for i in range(len(wili)):
        hot_idx = np.sum(wili[i] >= bins) - 1
        y[i, hot_idx] = 1

    return y

def get_merged_features(components, feature_functions):
    """
    Return a single matrix of distributions transformed to features
    """

    feature_blocks = []

    for fn in feature_functions:
        model_blocks = [
            np.array([fn(dist) for dist in comp]) for comp in components
        ]
        for im in range(len(model_blocks)):
            if len(model_blocks[im].shape) == 1:
                model_blocks[im] = np.expand_dims(model_blocks[im], axis=1)

        feature_blocks.append(np.concatenate(model_blocks, axis=1))

    return np.concatenate(feature_blocks, axis=1)
