"""
Utilities for working with distributions
"""

import numpy as np
import losses
import keras.backend as K
from typing import List
from scipy.stats import norm
from warnings import warn


BINS = {
    "wili": np.linspace(0, 12.9, 130),
    "peak_wk": np.arange(0, 33),
    "onset_wk": np.arange(0, 34)
}


def smooth_dists(dists: np.ndarray, window_len: int, window: str):
    """
    Smooth given dists
    """

    if window_len < 3:
        return dists
    if window == "flat":
        w = np.ones(window_len)
    else:
        w = eval("np." + window + "(window_len)")

    out = np.zeros_like(dists)
    for i in range(out.shape[0]):
        out[i] = np.convolve(w / w.sum(), dists[i], mode="same")
    return out


def dist_mean(dist, bins=BINS["wili"]):
    """
    Return the mean of distribution using default wili bins
    (skipping bin 13-100)
    """

    return np.sum(bins * dist)


def dist_max(dist, bins=BINS["wili"]):
    """
    Return value for max bin
    """

    return bins[np.argmax(dist)]


def dist_std(dist, bins=BINS["wili"]):
    """
    Return standard deviation
    """

    mean = dist_mean(dist, bins)
    var = np.sum(((bins - mean)**2) * dist)
    return np.sqrt(var)


def dist_median(dist, bins=BINS["wili"]):
    return np.max((np.cumsum(dist) < 0.5) * bins)


def dist_quartiles(dist, bins=BINS["wili"]):
    """
    Return quartiles division points
    """

    return np.array(
        [np.max((np.cumsum(dist) < i) * bins) for i in [0.25, 0.5, 0.75]])


def mdn_params_to_dists(params, bins=BINS["wili"]):
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


def actual_to_one_hot(wili, bins=BINS["wili"]):
    """
    Wili values to one hot encoded bins
    """

    y = np.zeros((wili.shape[0], bins.shape[0]))

    indices = np.digitize(wili, bins, right=True)
    if indices.max() == bins.shape[0]:
        warn("There are values hitting the upper limit in actual_to_one_hot")
        indices[indices == bins.shape[0]] = bins.shape[0] - 1
    for i in range(len(wili)):
        y[i, indices[i]] = 1

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


def get_2d_features(components):
    """
    Return image like 2d features for input in convolution models.
    Input is a list of n items of shape (batch_size, bins), output is a matrix
    of shape (batch_size, bins, n)
    """

    return np.array(components).transpose([1, 2, 0])


def shift_dists(dists,
                shift_values,
                bins=BINS["wili"]):
    """
    Shift the distributions by a value and renormalize to make it sum to one
    """

    # Convert shift values to number of bins
    bin_shift = [int(sv / (bins[1] - bins[0])) for sv in shift_values]

    output = np.zeros_like(dists)

    for i in range(dists.shape[0]):
        shift = bin_shift[i]
        output[i, :] = np.roll(dists[i, :], shift)
        # Don't circle around
        # if shift > 0:
        #     output[i, :shift] = np.zeros((shift, ))
        # elif shift < 0:
        #     output[i, -shift:] = np.zeros((shift, ))

    # output /= output.sum(axis=0) + K.epsilon()
    return output


def weighted_ensemble(dists: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Return weighted ensemble
    """

    return np.sum([d * w for d, w in zip(dists, weights)], axis=0)


def mean_ensemble(dists):
    """
    Return mean of dists. Works as mean ensemble model.
    """

    return weighted_ensemble(dists, np.ones(len(dists)) / len(dists))


def prod_ensemble(dists):
    """
    Return prod of dists. Works as product ensemble model.
    """

    log_dists = [np.log(dist + K.epsilon()) for dist in dists]
    return np.exp(mean_ensemble(log_dists))


def score_predictions(Xs: List[np.ndarray], y: np.ndarray) -> np.ndarray:
    """
    Return score matrix for the predictions
    """

    if Xs[0].shape[1] == 130:
        # This is weekly target
        # NOTE: We are skipping last bin [13.0, 100.0]
        product = np.stack([
            np.multiply(actual_to_one_hot(y), X).sum(axis=1) for X in Xs
        ], axis=1)
    elif Xs[0].shape[1] == 34:
        # This is onset week target
        product = np.stack([
            np.multiply(actual_to_one_hot(y, bins=BINS["onset_wk"]), X).sum(axis=1) for X in Xs
        ], axis=1)
    elif Xs[0].shape[1] == 33:
        # This is peak week target
        product = np.stack([
            np.multiply(actual_to_one_hot(y, bins=BINS["peak_wk"]), X).sum(axis=1) for X in Xs
        ], axis=1)
    else:
        raise Exception(f"Target type not understood. Shape given {Xs[0].shape}")

    return np.log(product)
