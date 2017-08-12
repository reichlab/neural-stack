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

    def get(self, data_identifier, region_identifier=None, epiweek_range=None):
        """
        Return data for asked data_identifier along with index

        Parameters
        ----------
        data_identifier : str | int
            Identifier for the target to load
        region_identifier : str | None
            Short region code (nat, hhs2 ...) or None for all regions
        epiweek_range : List[int]
            List of two ints representing the range of epiweeks (inclusive) to get data for
        """

        data = np.loadtxt(os.path.join(self.root_path, f"{data_identifier}.np.gz"))

        # All true selection
        selection = self.index["epiweek"] > 0
        narrowing = False
        if region_identifier is not None:
            selection = selection & (self.index["region"] == region_identifier)
            narrowing = True

        if epiweek_range is not None:
            selection = selection & (self.index["epiweek"] <= epiweek_range[1]) & (self.index["epiweek"] >= epiweek_range[0])
            narrowing = True

        if narrowing:
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


def filter_common_indices(*indices):
    """
    Return a list of integers for each of the indices such that slicing using
    these lists gives us dataframes matching the epiweek and regions in all the
    indices asked.
    """

    merge_on = ["epiweek", "region"]

    assert len(indices) > 1, "At least two indices needed"
    merged = pd.merge(indices[0].reset_index(), indices[1].reset_index(), on=merge_on)
    for i in range(2, len(indices)):
        merged = pd.merge(merged, indices[i].reset_index(), on=merge_on)

    # Return just the numbers
    return list(merged.drop(merge_on, axis=1).values.T)


def get_week_ahead_training_data(week_ahead, region_identifier, actual_data_loader, component_data_loaders):
    """
    Return well formed X's and y's for asked week and region

    Parameters
    -----------
    week_ahead : int
        A positive value of week ahead number, e.g. 1 if we are predicting one
        week ahead
    region_identifier : str
        A string representation for region (like "nat") or None if all data is
        needed
    actual_data_loader : ActualDataLoader
    component_data_loaders : List[ComponentDataLoader]
    """

    actual_idx, actual_data = actual_data_loader.get(week_shift=week_ahead, region_identifier=region_identifier)
    component_idx_data = [
        component_data_loader.get(week_ahead, region_identifier=region_identifier)
        for component_data_loader in component_data_loaders
    ]

    filter_indices = filter_common_indices(*[actual_idx, *[c[0] for c in component_idx_data]])

    y = actual_data[filter_indices[0]]

    # NOTE: Skipping the last bin
    Xs = []
    for i in range(len(component_idx_data)):
        Xs.append(
            np.exp(component_idx_data[i][1][filter_indices[i + 1]][:, :-1])
        )

    return y, Xs
