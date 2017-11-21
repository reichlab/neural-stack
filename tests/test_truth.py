"""
Test for true targets
"""

import sys
sys.path.append("./src")

import utils.data as udata
import utils.misc as u
import pandas as pd
import numpy as np


def test_peak():
    adl = udata.ActualDataLoader("./data")
    aidx, adata = adl.get()

    target = udata.Target("peak")
    y, Xs, yi = target.get_training_data(adl, [], None, 222222)

    assert all(aidx["epiweek"] == [i[0] for i in yi])
    assert all(aidx["region"] == [i[1] for i in yi])
    assert all(adata <= y)


def test_peak_wk():
    adl = udata.ActualDataLoader("./data")
    aidx, adata = adl.get()

    target = udata.Target("peak_wk")
    y, Xs, yi = target.get_training_data(adl, [], None, 222222)

    assert all(aidx["epiweek"] == [i[0] for i in yi])
    assert all(aidx["region"] == [i[1] for i in yi])

    assert all([
        u.epiweek_to_season(a) == u.epiweek_to_season(b)
        for a, b in zip([i[0] for i in yi], aidx["epiweek"])
    ])


def test_onset_wk():
    adl = udata.ActualDataLoader("./data")
    aidx, adata = adl.get()

    target = udata.Target("onset_wk")
    y, Xs, yi = target.get_training_data(adl, [], None, 222222)

    # Check if all onset weeks of same season are the same
    seasons = [u.epiweek_to_season(i[0]) for i in yi]
    df = pd.DataFrame({
        "epiweek": [i[0] for i in yi],
        "season": [u.epiweek_to_season(i[0]) for i in yi],
        "region": [i[1] for i in yi],
        "onset_wk": y
    })

    for name, group in df.groupby(["season", "region"]):
        assert group["onset_wk"].isnull().all() or len(set(group["onset_wk"])) == 1

    # Random checks
    onset_map = [
        [1997, "nat", 7],
        [2007, "nat", 12]
    ]

    for item in onset_map:
        predicted_onset = df[(df["season"] == item[0]) & (df["region"] == item[1])].iloc[0, :]["onset_wk"]
        assert int(predicted_onset) == item[-1]
