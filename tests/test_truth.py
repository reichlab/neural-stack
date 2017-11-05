"""
Test for true targets
"""

import src.utils.data as udata
import numpy as np
from operator import eq


def test_peak():
    adl = udata.ActualDataLoader("./data")
    aidx, adata = adl.get()
    y, Xs, yi = udata.get_seasonal_training_data("peak", None, adl, [])

    assert all(aidx["epiweek"] == [i[0] for i in yi])
    assert all(aidx["region"] == [i[1] for i in yi])
    assert all(adata <= y)


def test_peak_wk():
    adl = udata.ActualDataLoader("./data")
    aidx, adata = adl.get()
    y, Xs, yi = udata.get_seasonal_training_data("peak_wk", None, adl, [])

    assert all(aidx["epiweek"] == [i[0] for i in yi])
    assert all(aidx["region"] == [i[1] for i in yi])

    assert all([
        udata.epiweek_to_season(a) == udata.epiweek_to_season(b)
        for a, b in zip(y, aidx["epiweek"])
    ])
