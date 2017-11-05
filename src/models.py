"""
Models used in notebooks
"""

import numpy as np
from keras.layers import (Activation, Convolution1D, Convolution2D, Dense,
                          Dropout, Embedding, Flatten, Merge)
from keras.models import Sequential
from keras.regularizers import l2


def dem(mat, weights=None, epsilon=None):
    """
    Run the degenerate EM algorithm on given data. Return a set of weights for
    each model. Code replicates the method implemented in epiforecast-R package
    here https://github.com/cmu-delphi/epiforecast-R/blob/master/epiforecast/R/ensemble.R

    Parameters
    ----------
    mat : np.ndarray
        Shape (n_obs, n_models). Probabilities from n_models models for n_obs
        observations.
    weights : np.ndarray
        Initial weights
    epsilon : float
        Tolerance value
    """

    if weights is None:
        weights = np.ones(mat.shape[1]) / mat.shape[1]

    if not epsilon:
        epsilon = np.sqrt(np.finfo(float).eps)

    w_mat = mat * weights
    marginals = np.sum(w_mat, axis=1)
    log_marginal = np.mean(np.log(marginals))

    if np.isneginf(log_marginal):
        raise ValueError("All methods assigned a probability of 0 to at least one observed event.")
    else:
        while True:
            prev_log_marginal = log_marginal
            weights = np.mean(w_mat.T / marginals, axis=1)
            w_mat = mat * weights
            marginals = np.sum(w_mat, axis=1)
            log_marginal = np.mean(np.log(marginals))

            if log_marginal + epsilon < prev_log_marginal:
                raise ValueError("Log marginal less than prev_log_marginal")
            marginal_diff = log_marginal - prev_log_marginal
            if (marginal_diff <= epsilon) or ((marginal_diff / -log_marginal) <= epsilon):
                break
    return weights


def conv1D_distribution(n_models,
                        n_bins,
                        week_embedding_size,
                        week_embedding_matrix=None):
    """
    One dimensional conv model over input distribution to give an output
    distribution

    Merges two branches
    - predictions : (batch_size, n_bins, n_models)
    - weeks : (batch_size, 1)

    Parameters
    ----------
    n_models : int
        Number of models (predictive distributions)
    n_bins : int
        Number of bins in the prediction distribution
    week_embedding_size : int
        Embedding vector size for week
    week_embedding_matrix : np.ndarray
        Embedding matrix to use as initial weight
    """

    preds = Sequential()

    preds.add(
        Convolution1D(
            64, 5, border_mode="same", input_shape=(n_bins, n_models)))

    preds.add(Convolution1D(20, 5, border_mode="same"))
    preds.add(Flatten())
    preds.add(Dense(30, W_regularizer=l2(0.01)))
    preds.add(Activation("tanh"))

    weeks = Sequential()
    # Encoding all the weeks possible (not just the ones from 10 to 30)
    if week_embedding_matrix is not None:
        weeks.add(
            Embedding(
                54,
                week_embedding_matrix.shape[1],
                input_length=1,
                weights=[week_embedding_matrix]))
    else:
        weeks.add(Embedding(54, week_embedding_size, input_length=1))
    weeks.add(Flatten())
    weeks.add(Activation("tanh"))
    weeks.add(Dense(10, W_regularizer=l2(0.01)))

    merged = Sequential()
    merged.add(Merge([preds, weeks], mode="concat", concat_axis=1))
    merged.add(Dense(30, W_regularizer=l2(0.01)))
    merged.add(Activation("relu"))
    merged.add(Dense(n_bins))
    merged.add(Activation("softmax"))

    return merged
