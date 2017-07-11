"""
Helper module for working with data
"""

import numpy as np
import os
import pandas as pd
import pymmwr


class ComponentDataLoader:
    """
    Data loader for component models
    """

    def __init__(self, data_dir: str, model_identifier: str) -> None:
        self.root_path = os.path.join(data_dir, "processed", "components", model_identifier)
        self.index = pd.read_csv(os.path.join(self.root_path, "index.csv"))

    def get(self, data_identifier: str, region_identifier=None):
        """
        Return data for asked data_identifier along with index
        """

        data = np.loadtxt(os.path.join(self.root_path, f"{data_identifier}.np.gz"))

        if region_identifier:
            selection = self.index["region"] == region_identifier
            return [self.index[selection].reset_index(drop=True), data[selection]]
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

    def get(self, week_shift=None, region_identifier=None):
        """
        Return index and data for given region. Return all if the region is
        None. week_shift of x gives wili of 'wk + x' for each epiweek 'wk'.
        """

        # Subset by region
        if region_identifier:
            selection = self.index["region"] == region_identifier
            index = self.index[selection].reset_index(drop=True)
            wili = self._df[selection]["wili"].values
        else:
            index = self.index
            wili = self._df["wili"].values

        # Handle week shifts
        if week_shift:
            shifted_selection = []
            shifted_data_indices = []
            for idx, row in index.iterrows():
                shifted = pymmwr.mmwr_week_with_delta(row["epiweek"] // 100, row["epiweek"] % 100, week_shift)
                shifted_epiweek = shifted["year"] * 100 + shifted["week"]
                try:
                    shifted_next_row = index.iloc[idx + week_shift, :]
                except IndexError:
                    shifted_selection.append(False)
                    continue

                if shifted_next_row["region"] == row["region"] and shifted_next_row["epiweek"] == shifted_epiweek:
                    shifted_selection.append(True)
                    shifted_data_indices.append(idx + week_shift)
                else:
                    shifted_selection.append(False)
            return [index[shifted_selection].reset_index(drop=True), wili[shifted_data_indices]]

        else:
            return [index, wili]
