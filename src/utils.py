"""
Utilities for working with distributions and similar stuff
"""

import keras.backend as K
import numpy as np


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

    return [
        np.max((np.cumsum(dist) < i) * bins)
        for i in [0.25, 0.5, 0.75]
    ]
