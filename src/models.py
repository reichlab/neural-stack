"""
Models used in notebooks
"""

import abc

from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, Flatten, Merge
from keras.layers import Convolution1D
from keras.regularizers import l2


class ModelMeta(abc.ABC):
    """
    Abstract class enforcing model design
    """

    @abc.abstractmethod
    def fit():
        """
        Use this to train the model
        """

    @abc.abstractmethod
    def predict():
        """
        Use this to take out predictions
        """


class SimpleDense(ModelMeta):
    """
    Basic dense model taking embedding of input
    """

    def __init__(self, input_length, output_length):
        model = Sequential()

        model.add(Embedding(54, 10, input_length=input_length))
        model.add(Flatten())
        model.add(Activation("tanh"))
        model.add(Dense(10, W_regularizer=l2(0.08)))
        model.add(Activation("tanh"))
        model.add(Dense(output_length, W_regularizer=l2(0.01)))

        model.compile(loss="mse", optimizer="rmsprop")

        self.model = model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args):
        return self.model.predict(*args)


class Conv1DDistribution(ModelMeta):
    """
    One dimensional conv model over input distribution to give an output
    distribution

    Merges two branches
    - predictions : (batch_size, n_bins, n_models)
    - weeks : (batch_size, 1)
    """

    def __init__(self, n_models, n_bins, week_embedding_size):
        """
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
                32, 5, border_mode="same", input_shape=(n_bins, n_models)))

        preds.add(Convolution1D(16, 3, border_mode="same"))
        preds.add(Flatten())
        preds.add(Dense(50))
        preds.add(Activation("tanh"))

        weeks = Sequential()
        # Encoding all the weeks possible (not just the ones from 10 to 30)
        weeks.add(Embedding(54, week_embedding_size, input_length=1))
        weeks.add(Flatten())
        weeks.add(Activation("tanh"))
        weeks.add(Dense(10, W_regularizer=l2(0.01)))

        merged = Sequential()
        merged.add(Merge([preds, weeks], mode="concat", concat_axis=1))
        merged.add(Dense(50))
        merged.add(Activation("relu"))
        merged.add(Dense(n_bins))
        merged.add(Activation("softmax"))

        merged.compile(loss="categorical_crossentropy", optimizer="rmsprop")

        self.model = merged
        self.n_bins = n_bins
        self.n_models = n_models
        self.week_embedding_size = week_embedding_size

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args):
        return self.model.predict(*args)
