"""
Helper module for working with data
"""

import numpy as np
import pandas as pd
import os
import pymmwr
from collections import namedtuple
from typing import List
from utils import misc as u


Component = namedtuple("Component", ["name", "loader"])


class ComponentDataLoader:
    """
    Data loader for component models
    """

    def __init__(self, exp_dir: str, model_identifier: str) -> None:
        self.root_path = os.path.join(exp_dir, model_identifier)
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


def get_components(data_dir: str, names: List[str]) -> List[Component]:
    """
    Return simple wrapper over component loader using provided names
    """

    return [Component(name=name, loader=ComponentDataLoader(data_dir, name)) for name in names]


class ActualDataLoader:
    """
    Data loader for actual data
    """

    def __init__(self, data_dir: str) -> None:
        self.root_path = os.path.join(data_dir, "processed")
        self._df = pd.read_csv(os.path.join(self.root_path, "actual.csv"))
        self.baseline = pd.read_csv(os.path.join(self.root_path, "baseline.csv"))
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


def filter_common_indices(indices):
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


def get_seasonal_training_data(target, region_identifier, actual_data_loader, component_data_loaders):
    """
    Return well formed y, Xs and yi for asked week and region
    """

    actual_idx, actual_data = actual_data_loader.get(region_identifier=region_identifier)
    component_idx_data = [
        component_data_loader.get(target, region_identifier=region_identifier)
        for component_data_loader in component_data_loaders
    ]

    filter_indices = filter_common_indices([actual_idx, *[c[0] for c in component_idx_data]])

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
    if target == "peak":
        y = peaks_df["peak"].values
    elif target == "peak_wk":
        y = np.array([u.epiweek_to_model_week(ew) for ew in peaks_df["peak_wk"].values])
    elif target == "onset_wk":
        onset_wks = {
            "region": [],
            "season": [],
            "onset_wk": []
        }
        for name, group in peaks_df.groupby(["season", "region"]):
            season, region = name
            onset_wks["season"].append(season)
            onset_wks["region"].append(region)
            onset_wks["onset_wk"].append(_get_onset_wk(group))
        onset_wks = pd.DataFrame(onset_wks)

        onset_output = peaks_df.merge(onset_wks, on=["season", "region"]).sort_values("order")["onset_wk"].values
        y = np.array([u.epiweek_to_model_week(ew) for ew in onset_output])
    else:
        raise Exception(f"Unknown target {target}")

    # NOTE:
    # Expecting 131 bins for peak
    # 34 for onset_wk (one for onset None)
    # 33 for peak_wk
    Xs = []
    for i in range(len(component_idx_data)):
        if target == "peak":
            # Skip last bin [13.0, 100]
            Xs.append(component_idx_data[i][1][filter_indices[i + 1]][:, :-1])
        else:
            Xs.append(component_idx_data[i][1][filter_indices[i + 1]])

    return y, Xs, peaks_df[["epiweek", "region"]].as_matrix()


def get_week_ahead_training_data(week_ahead, region_identifier, actual_data_loader, component_data_loaders):
    """
    Return well formed y, Xs and yi for asked week and region

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

    filter_indices = filter_common_indices([actual_idx, *[c[0] for c in component_idx_data]])

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
            return get_week_ahead_training_data

        else:
            return get_seasonal_training_data

    def _get_all_data(self, actual_dl, components, region):
        return self.getter_fn(
            self._name, region,
            actual_dl, [c.loader for c in components]
        )

    def get_training_data(self, actual_dl, components, region, split_thresh):
        """
        Return training y, Xs, yi for target and all regions
        """

        y, Xs, yi = self._get_all_data(actual_dl, components, region)
        train_indices = yi[:, 0] < split_thresh
        return y[train_indices], [X[train_indices] for X in Xs], yi[train_indices]

    def get_testing_data(self, actual_dl, components, region, split_thresh):
        """
        Return testing y, Xs, yi for target and all regions
        """

        y, Xs, yi = self._get_all_data(actual_dl, components, region)
        test_indices = yi[:, 0] >= split_thresh
        return y[test_indices], [X[test_indices] for X in Xs], yi[test_indices]
