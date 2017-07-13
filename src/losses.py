"""
Custom loss functions
"""

import keras.backend as K
import numpy as np


def separate_mdn_params(params, n_mix):
    """
    Separate single network output to 3 mixture components

    Parameters
    -----------
    params : tensor (batch_size, n_mix * 3)
        Params from the network
    n_mix : int
        Number of mixture
    """

    # Parameters are arranged in sequence. First means, then sigmas and then
    # mixture weights
    mu = params[:, :n_mix]
    sigma = K.relu(params[:, n_mix:2 * n_mix])
    w = params[:, 2 * n_mix:]
    w = K.softmax(w)

    return mu, sigma, w


def mdn_loss(y, params, n_mix=None):
    """
    Loss function for mixture density network. This sometimes gets into
    numerical issues. Will need some tweaking.

    Parameters:
    ------------
    y : tensor (batch_size, )
        Actual value for fitting
    params : tensor (batch_size, n_mix * 3)
        The output from the network. The output layer provides 3 nodes for
        each of the n_mix mixtures.
    n_mix : int
        Number of mixture in the model
    """

    def normal(y, mu, sigma):
        """
        Return probability at y for given mu-sigma distribution.
        This uses the direct formula for normal distribution along with some
        tensor reshaping.

        Parameters
        -----------
        y : tensor (batch_size, )
            Values to find probabilities at
        mu : tensor (batch_size, n_mix)
            Mean for each of the mixture
        sigma : tensor (batch_size, n_mix)
            Standard deviation for each of the mixture
        """

        sigma_ep = sigma + K.epsilon()

        out = -(K.reshape(y, (-1, 1)) - mu) ** 2
        out /= (2 * sigma_ep ** 2)
        out = K.exp(out)
        out /= sigma_ep * np.sqrt(2 * np.pi)
        return out

    mu, sigma, w = separate_mdn_params(params, n_mix)

    out = normal(y, mu, sigma)
    out *= w
    out = K.sum(out, axis=-1, keepdims=True)

    return K.mean(-K.log(out + K.epsilon()), axis=-1)
