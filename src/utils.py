"""
Utilities
"""

import keras.backend as K
import numpy as np


def distributions_to_value(y_pred):
    """
    Return maxed numbers from prediction distribution
    """

    return np.argmax(y_pred, axis=1) / 10
