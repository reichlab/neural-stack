"""
Tests for component data directory
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path


# Numpy files expected in the model dir
NP_FILES = [
    "1.np.gz",
    "2.np.gz",
    "3.np.gz",
    "4.np.gz",
    "onset_wk.np.gz",
    "peak.np.gz",
    "peak_wk.np.gz"
]


def get_component_dirs():
    """
    Return model dir paths
    """

    return [x for x in Path("./data/processed/components").iterdir() if x.is_dir()]


def get_index_rows(component_dir) -> int:
    """
    Return number of rows in the model index
    """

    return pd.read_csv(component_dir.joinpath("index.csv")).shape[0]


def get_np_rows(np_file) -> int:
    return np.loadtxt(np_file).shape[0]


def get_np_cols(np_file) -> int:
    return np.loadtxt(np_file).shape[1]


def test_files():
    for component_dir in get_component_dirs():
        for f in [*NP_FILES, "index.csv"]:
            assert component_dir.joinpath(f).exists()


def test_shape_consistency():
    """
    Test if number of rows match in all the files
    """

    for component_dir in get_component_dirs():
        nrows = get_index_rows(component_dir)
        for f in NP_FILES:
            assert nrows == get_np_rows(component_dir.joinpath(f))


def test_np_cols():
    """
    Test if the numpy files have expected cols
    """

    ncols = {
        "1.np.gz": 131,
        "2.np.gz": 131,
        "3.np.gz": 131,
        "4.np.gz": 131,
        "onset_wk.np.gz": 34,
        "peak.np.gz": 131,
        "peak_wk.np.gz": 33
    }

    for component_dir in get_component_dirs():
        for f in NP_FILES:
            assert get_np_cols(component_dir.joinpath(f)) == ncols[f]
