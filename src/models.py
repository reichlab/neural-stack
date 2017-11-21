"""
Models used in notebooks
"""

import numpy as np


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
