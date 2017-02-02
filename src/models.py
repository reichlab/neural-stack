"""
Models used in notebooks
"""

from keras.layers import (Activation, Convolution1D, Convolution2D, Dense,
                          Embedding, Flatten, Merge)
from keras.models import Sequential
from keras.regularizers import l2


def simple_dense(input_length, output_length):
    """
    Basic dense model taking embedding of input
    """

    model = Sequential()
    model.add(Embedding(54, 10, input_length=input_length))
    model.add(Flatten())
    model.add(Activation("tanh"))
    model.add(Dense(10, W_regularizer=l2(0.08)))
    model.add(Activation("tanh"))
    model.add(Dense(output_length, W_regularizer=l2(0.01)))
    return model


def simple_dense_direct(n_models, week_embedding_size):
    """
    Take log scores and week number to give actual point prediction directly

    Parameters
    ----------
    n_models : int
        Number of model log scores
    week_embedding_size : int
        Embedding vector size for week
    """

    log_scores = Sequential()
    log_scores.add(Dense(10, input_dim=n_models))
    log_scores.add(Activation("tanh"))
    log_scores.add(Dense(week_embedding_size))

    weeks = Sequential()
    # Encoding all the weeks possible (not just the ones from 10 to 30)
    weeks.add(Embedding(54, week_embedding_size, input_length=1))
    weeks.add(Flatten())
    weeks.add(Activation("tanh"))
    weeks.add(Dense(10, W_regularizer=l2(0.01)))

    merged = Sequential()
    merged.add(Merge([log_scores, weeks], mode="concat", concat_axis=-1))
    merged.add(Dense(20))
    merged.add(Activation("relu"))
    merged.add(Dense(10))
    merged.add(Activation("relu"))
    merged.add(Dense(1))

    return merged


def conv1D_distribution(n_models, n_bins, week_embedding_size):
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
    weeks.add(Embedding(54, week_embedding_size, input_length=1))
    weeks.add(Flatten())
    weeks.add(Activation("tanh"))
    weeks.add(Dense(10, W_regularizer=l2(0.01)))

    merged = Sequential()
    merged.add(Merge([preds, weeks], mode="concat", concat_axis=1))
    merged.add(Dense(50, W_regularizer=l2(0.01)))
    merged.add(Activation("relu"))
    merged.add(Dense(n_bins))
    merged.add(Activation("softmax"))

    return merged


def conv2D_distribution(n_models, n_bins, week_embedding_size):
    """
    Two dimensional conv model over input distribution to give an output
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
    """

    preds = Sequential()

    preds.add(
        Convolution2D(
            64, 3, 3, border_mode="same", input_shape=(1, n_bins, n_models)))

    preds.add(Convolution2D(32, 3, 3, border_mode="same"))
    preds.add(Flatten())
    preds.add(Dense(30, W_regularizer=l2(0.01)))
    preds.add(Activation("tanh"))

    weeks = Sequential()
    # Encoding all the weeks possible (not just the ones from 10 to 30)
    weeks.add(Embedding(54, week_embedding_size, input_length=1))
    weeks.add(Flatten())
    weeks.add(Activation("tanh"))
    weeks.add(Dense(10, W_regularizer=l2(0.01)))

    merged = Sequential()
    merged.add(Merge([preds, weeks], mode="concat", concat_axis=1))
    merged.add(Dense(50, W_regularizer=l2(0.01)))
    merged.add(Activation("relu"))
    merged.add(Dense(n_bins))
    merged.add(Activation("softmax"))

    return merged
