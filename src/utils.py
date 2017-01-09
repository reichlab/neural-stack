"""
Utilities
"""

import keras.backend as K


def prediction_score(y_true, y_pred, temperature=1.0):
    """
    Return log score of predictions using provided temperature value
    """

    y_pred = K.log(y_pred) / temperature
    y_pred = K.exp(y_pred) / K.sum(K.exp(y_pred), axis=-1, keepdims=True)

    return K.categorical_crossentropy(y_pred, y_true)