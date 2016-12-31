"""
Models used in notebooks
"""


from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, Flatten
from keras.regularizers import l2


class SimpleDense:
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
