"""
Helper module for working with data
"""

import numpy as np
import pandas as pd
import os
import pymmwr
from collections import namedtuple
from typing import List, Tuple
from utils import misc as u
from functools import lru_cache


class Component:
    """
    Data loader for component models
    """

    def __init__(self, exp_dir: str, name: str) -> None:
        self.name = name
        self.root_path = os.path.join(exp_dir, name)
        self.index = pd.read_csv(os.path.join(self.root_path, "index.csv"))

    @lru_cache(maxsize=128)
    def get(self, target_name, region=None, epiweek_range=None):
        """
        Return data for asked target_name along with index

        Parameters
        ----------
        target_name : str | int
            Identifier for the target to load
        region : str | None
            Short region code (nat, hhs2 ...) or None for all regions
        epiweek_range : Tuple[int]
            List of two ints representing the range of epiweeks (inclusive) to get data for
        """

        data = np.loadtxt(os.path.join(self.root_path, f"{target_name}.np.gz"))

        # All true selection
        selection = self.index["epiweek"] > 0
        narrowing = False
        if region is not None:
            selection = selection & (self.index["region"] == region)
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
        self.baseline = pd.read_csv(os.path.join(self.root_path, "baseline.csv"))
        self.index = self._df[["epiweek", "region"]]

    @lru_cache(maxsize=128)
    def get(self, week_shift=None, region=None):
        """
        Return index and data for given region. Return all if the region is
        None. week_shift of x gives wili of 'wk + x' for each epiweek 'wk'.
        """

        # Subset by region
        if region is not None:
            selection = self.index["region"] == region
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


def _filter_common_indices(indices):
    """
    Return a list of integers for each of the indices such that slicing using
    these lists gives us dataframes matching the epiweek and regions in all the
    indices asked.
    """

    merge_on = ["epiweek", "region"]

    if len(indices) == 1:
        return [indices[0].index.values]

    merged = pd.merge(indices[0].reset_index(), indices[1].reset_index(), on=merge_on)
    for i in range(2, len(indices)):
        merged = pd.merge(merged, indices[i].reset_index(), on=merge_on)

    # Return just the numbers
    return list(merged.drop(merge_on, axis=1).values.T)


def _get_seasonal_training_data(target_name: str, region, actual_data_loader: ActualDataLoader, components: List[Component]):
    """
    Return well formed y, Xs and yi for asked week and region
    """

    actual_idx, actual_data = actual_data_loader.get(region=region)
    component_idx_data = [c.get(target_name, region=region) for c in components]

    filter_indices = _filter_common_indices([actual_idx, *[c[0] for c in component_idx_data]])

    true_df = actual_idx.copy().iloc[filter_indices[0], :]
    true_df["wili"] = actual_data[filter_indices[0]]
    true_df["season"] = true_df.apply(lambda row: u.epiweek_to_season(row["epiweek"]), axis=1)
    true_df["order"] = np.arange(0, true_df.shape[0])

    # Calculate peak week and value maps
    peaks_df = true_df.sort_values("wili", ascending=False).drop_duplicates(["season", "region"])
    peaks_df = true_df.merge(peaks_df, on=["season", "region"], suffixes=("", "_x"))
    peaks_df = peaks_df.rename(columns={"epiweek_x": "peak_wk", "wili_x": "peak"})
    peaks_df = peaks_df.sort_values("order")
    peaks_df = peaks_df.drop("order_x", axis=1)
    peaks_df = peaks_df.merge(actual_data_loader.baseline, on=["season", "region"])

    def _get_onset_wk(subset):
        """
        Return onset week for region, season chunk
        """

        try:
            return subset[(subset["wili"] - subset["baseline"]) > 0].sort_values("epiweek").iloc[0, :]["epiweek"]
        except IndexError:
            return None

    y = []
    if target_name == "peak":
        y = peaks_df["peak"].values
    elif target_name == "peak_wk":
        y = np.array([u.epiweek_to_model_week(ew) for ew in peaks_df["peak_wk"].values])
    elif target_name == "onset_wk":
        onset_wks = {
            "region": [],
            "season": [],
            "onset_wk": []
        }
        for name, group in peaks_df.groupby(["season", "region"]):
            season, reg = name
            onset_wks["season"].append(season)
            onset_wks["region"].append(reg)
            onset_wks["onset_wk"].append(_get_onset_wk(group))
        onset_wks = pd.DataFrame(onset_wks)

        onset_output = peaks_df.merge(onset_wks, on=["season", "region"]).sort_values("order")["onset_wk"].values
        y = np.array([u.epiweek_to_model_week(ew) for ew in onset_output])
    else:
        raise Exception(f"Unknown target {target_name}")

    # NOTE:
    # Expecting 131 bins for peak
    # 34 for onset_wk (one for onset None)
    # 33 for peak_wk
    Xs = []
    for i in range(len(component_idx_data)):
        if target_name == "peak":
            # Skip last bin [13.0, 100]
            Xs.append(component_idx_data[i][1][filter_indices[i + 1]][:, :-1])
        else:
            Xs.append(component_idx_data[i][1][filter_indices[i + 1]])

    return y, Xs, peaks_df[["epiweek", "region"]].as_matrix()


def _get_week_ahead_training_data(week_ahead: int, region, actual_data_loader: ActualDataLoader, components: List[Component]):
    """
    Return well formed y, Xs and yi for asked week and region
    """

    actual_idx, actual_data = actual_data_loader.get(week_shift=week_ahead, region=region)
    component_idx_data = [c.get(week_ahead, region=region) for c in components]

    filter_indices = _filter_common_indices([actual_idx, *[c[0] for c in component_idx_data]])

    y = actual_data[filter_indices[0]]

    # NOTE: Skipping the last bin
    Xs = []
    for i in range(len(component_idx_data)):
        Xs.append(component_idx_data[i][1][filter_indices[i + 1]][:, :-1])

    return y, Xs, actual_idx.as_matrix()[filter_indices[0]]


class Target:
    """
    Class collecting properties of a target
    """

    def __init__(self, name) -> None:
        self._name = name

    @property
    def name(self):
        """
        Convert to string to handle numerical targets
        """

        return str(self._name)

    @property
    def type(self):
        if self._name in range(1, 5):
            return "weekly"
        else:
            return "seasonal"

    @property
    def bins(self):
        if self._name in [1, 2, 3, 4, "peak"]:
            return np.linspace(0, 12.9, 130)
        elif self._name == "peak_wk":
            return np.arange(0, 33)
        elif self._name == "onset_wk":
            return np.arange(0, 34)

    @property
    def getter_fn(self):
        if self.type == "weekly":
            return _get_week_ahead_training_data

        else:
            return _get_seasonal_training_data

    def _get_all_data(self, actual_dl, components, region):
        return self.getter_fn(self._name, region, actual_dl, components)

    def get_training_data(self, actual_dl: ActualDataLoader, components: List[Component], region, split_thresh: int):
        """
        Return training y, Xs, yi for target and all regions
        """

        y, Xs, yi = self._get_all_data(actual_dl, components, region)
        train_indices = yi[:, 0] < split_thresh
        return y[train_indices], [X[train_indices] for X in Xs], yi[train_indices]

    def get_testing_data(self, actual_dl: ActualDataLoader, components: List[Component], region, split_thresh: int):
        """
        Return testing y, Xs, yi for target and all regions
        """

        y, Xs, yi = self._get_all_data(actual_dl, components, region)
        test_indices = yi[:, 0] >= split_thresh
        return y[test_indices], [X[test_indices] for X in Xs], yi[test_indices]
