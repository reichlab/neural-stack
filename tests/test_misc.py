"""
Misc test functions
"""

import sys
sys.path.append("./src/")
import src.utils.data as udata
import src.utils.dists as udists
import src.utils.misc as u
import numpy as np

def test_model_week():
    for ew in [201434, 201354]:
        try:
            u.epiweek_to_model_week(ew)
            assert False
        except:
            assert True

    assert u.epiweek_to_model_week(200746) == 6
    assert u.epiweek_to_model_week(200816) == 28
    assert u.epiweek_to_model_week(201516) == 29

    assert u.epiweek_to_model_week(np.nan) == 33


def test_wili_to_one_hot():
    actual = np.array([0.0, 0.1, 0.6, 12.9, 2.3])

    one_hot = np.zeros((len(actual), 130))
    for idx, val in enumerate(actual):
        one_hot[idx, int(val * 10)] = 1

    assert np.allclose(udists.actual_to_one_hot(actual), one_hot)


def test_onset_wk_to_one_hot():
    actual = np.array([0, 23, 33, 2])

    one_hot = np.zeros((len(actual), 34))
    for idx, val in enumerate(actual):
        one_hot[idx, val] = 1

    assert np.allclose(udists.actual_to_one_hot(actual, bins=udists.BINS["onset_wk"]), one_hot)

def test_peak_wk_to_one_hot():
    actual = np.array([0, 23, 32, 2])

    one_hot = np.zeros((len(actual), 33))
    for idx, val in enumerate(actual):
        one_hot[idx, val] = 1

    assert np.allclose(udists.actual_to_one_hot(actual, bins=udists.BINS["peak_wk"]), one_hot)
