"""
Tests related to handling submission csvs
"""

import src.submission as sub
import numpy as np
from pathlib import Path


CSVS = [
    Path("./tests/data/week52.csv"),
    Path("./tests/data/week53-type1.csv"),
    Path("./tests/data/week53-type2.csv")
]


def get_expected_bin_order(target, week53=False):
    """
    Return expected bin order for given target. Handle week 53 case
    if week53 is True.
    """

    if target in [1, 2, 3, 4, "peak"]:
        return np.linspace(0, 13, 131)

    year_end = 53 if week53 else 52
    season_end = 19 if week53 else 20
    week_bins = list(range(40, year_end + 1)) + list(range(1, season_end + 1))

    if target == "onset_wk":
        return [*week_bins, np.nan]
    elif target == "peak_wk":
        return week_bins


def test_binorder():
    """
    Test if order returned while get_X-ing is correct
    """

    for csv in CSVS:
        week53 = csv.name.startswith("week53")
        s = sub.Submission(csv=csv)
        for region in sub.MAP_REGION:
            for target in sub.MAP_TARGET:
                _, bin_order = s.get_X(region, target)
                assert np.allclose(bin_order, get_expected_bin_order(target, week53), equal_nan=True)
