"""
Utilities for working with distributions and similar stuff
"""

import keras.backend as K
import numpy as np
import pandas as pd
import losses
import pymmwr
import os
import matplotlib.pyplot as plt
from functools import reduce
from tabulate import tabulate
from tqdm import tqdm
from scipy.stats import norm
from typing import Dict
from sklearn.model_selection import KFold


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


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
    var = np.sum(((bins - mean)**2) * dist)
    return np.sqrt(var)


def dist_median(dist, bins=np.linspace(0, 12.9, 130)):
    return np.max((np.cumsum(dist) < 0.5) * bins)


def dist_quartiles(dist, bins=np.linspace(0, 12.9, 130)):
    """
    Return quartiles division points
    """

    return np.array(
        [np.max((np.cumsum(dist) < i) * bins) for i in [0.25, 0.5, 0.75]])


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


def shift_distribution(distributions,
                       shift_values,
                       bins=np.linspace(0, 12.9, 130)):
    """
    Shift the distribution by a value and renormalize to make it sum to one
    """

    # Convert shift values to number of bins
    bin_shift = [int(sv / (bins[1] - bins[0])) for sv in shift_values]

    output = np.zeros_like(distributions)

    for i in range(distributions.shape[0]):
        shift = bin_shift[i]
        output[i, :] = np.roll(distributions[i, :], shift)
        # Don't circle around
        # if shift > 0:
        #     output[i, :shift] = np.zeros((shift, ))
        # elif shift < 0:
        #     output[i, -shift:] = np.zeros((shift, ))

    # output /= output.sum(axis=0) + K.epsilon()
    return output


def cv_train_kfold(gen_model, train_model, X, y, k=10):
    """
    Train the model using KFold cross validation.

    Parameters
    ----------
    gen_model : function
        Function that returns a newly created model when called
    train_model : function
        Function taking in model, train_data, val_data and performing actual
        training. It returns keras training history for the run.
    X : np.ndarray
        Numpy array representing the model input
    y : np.ndarray
        Numpy array representing the actual output
    k : int
        k value for the k-fold split
    """

    kf = KFold(n_splits=k, shuffle=True)

    histories = []
    for train_indices, val_indices in tqdm(kf.split(X)):
        model = gen_model()
        train_data = (X[train_indices], y[train_indices])
        val_data = (X[val_indices], y[val_indices])
        histories.append(train_model(model, train_data, val_data))

    return  [
        {
            "training_loss": history.history["loss"][-1],
            "validation_loss": history.history["val_loss"][-1],
            "history": history
        }
        for history in histories
    ]


def cv_train_loso(gen_model, train_model, X, y, yi):
    """
    Train the model using leave-one-season-out cross validation.

    Parameters
    ----------
    gen_model : function
        Function that returns a newly created model when called
    train_model : function
        Function taking in model, train_data, val_data and performing actual
        training. It returns keras training history for the run.
    X : np.ndarray
        Numpy array representing the model input
    y : np.ndarray
        Numpy array representing the actual output
    yi : np.ndarray
        Numpy array of (epiweek, region) indices
    """

    # yi to season indices
    def epiweek_to_season(ew):
        year, week = ew // 100, ew % 100
        if week <= 40:
            return year - 1
        else:
            return year

    seasons = [epiweek_to_season(i[0]) for i in yi]
    unique_seasons = list(set(seasons))

    histories = []
    for season in tqdm(unique_seasons):
        model = gen_model()
        train_indices = np.array([i != season for i in seasons])
        train_data = (X[train_indices], y[train_indices])
        val_data = (X[~train_indices], y[~train_indices])
        histories.append(train_model(model, train_data, val_data))

    cv_metadata = [
        {
            "training_loss": history.history["loss"][-1],
            "validation_loss": history.history["val_loss"][-1],
            "history": history
        }
        for history in histories
    ]

    return cv_metadata


def cv_report(cv_metadata):
    """
    Report the results of cross validation
    """

    lens = [len(it["history"].history["loss"]) for it in cv_metadata]
    losses = [it["training_loss"] for it in cv_metadata]
    val_losses = [it["validation_loss"] for it in cv_metadata]

    return pd.DataFrame({
        "epochs": lens + [np.mean(lens)],
        "train_loss": losses + [np.mean(losses)],
        "val_loss": val_losses + [np.mean(val_losses)]
    }, index=["it-" + str(i) for i in range(1, len(cv_metadata) + 1)] + ["mean"])


def cv_plot(cv_metadata):
    """
    Plot the training histories for cross validation
    """

    n = len(cv_metadata)

    cols = 2
    rows = (n // cols) + ((n % 2) * 1)

    f, axes = plt.subplots(rows, cols, figsize=(10, 15))

    for i in range(rows):
        for j in range(cols):
            axes[i][j].plot(cv_metadata[(i * cols) + j]["history"].history["loss"])
            axes[i][j].plot(cv_metadata[(i * cols) + j]["history"].history["val_loss"])


def mean_ensemble(dists):
    """
    Return mean of dists. Works as mean ensemble model.
    """

    return np.mean(dists, axis=0)


def prod_ensemble(dists):
    """
    Return prod of dists. Works as product ensemble model.
    """

    prod_dist = reduce(np.multiply, dists)
    prod_dist /= prod_dist.sum(axis=1, keepdims=True) + K.epsilon()
    return prod_dist


def save_exp_summary(model, cv_rep: pd.DataFrame, final_metadata: Dict, output_file: str):
    """
    Save a summary text for current experiment
    """

    with open(output_file, "w") as fp:
        fp.write("Model summary\n")
        fp.write("-------------\n")
        model.summary(print_fn=lambda line: fp.write(line + "\n"))
        fp.write("\n\n")
        fp.write("Cross validation\n")
        fp.write("----------------\n")
        fp.write(tabulate(cv_rep))
        fp.write("\n\n")
        fp.write("Final training\n")
        fp.write("--------------\n")
        fp.write(f"Epochs: {final_metadata['epochs']}\n")
        fp.write(f"Loss: {final_metadata['loss']}")
