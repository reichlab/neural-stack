"""
Helper module for working with data
"""

import numpy as np
import os
import pandas as pd


class ComponentDataLoader:
    """
    Data loader for component models
    """

    def __init__(self, data_dir: str, model_identifier: str) -> None:
        self.root_path = os.path.join(data_dir, "processed", "components", model_identifier)
        self.index = pd.read_csv(os.path.join(self.root_path, "index.csv"))

    def get(self, data_identifier: str, region_identifier = None):
        """
        Return data for asked data_identifier along with index
        """

        data = np.loadtxt(os.path.join(self.root_path, f"{data_identifier}.np.gz"))

        if region_identifier:
            selection = self.index["region"] == region_identifier
            return [self.index[selection], data[selection]]
        else:
            return [self.index, data]


class ActualDataLoader:
    """
    Data loader for actual data
    """

    def __init__(self, data_dir: str) -> None:
        self.root_path = os.path.join(data_dir, "processed")
        self._df = pd.read_csv(os.path.join(self.root_path, "actual.csv"))
        self.index = self._df[["epiweek", "region"]]

    def get(self, region_identifier = None):
        """
        Return a list of index and data for given region. Return all if the
        region is None
        """

        if region_identifier:
            selection = self.index["region"] == region_identifier
            return [self.index[selection], self._df[selection]["wili"].values]
        else:
            return [self.index, self._df["wili"].values]
