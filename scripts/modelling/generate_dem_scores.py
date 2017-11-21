"""
Generate scores from the degenerate em weights
"""

import sys
sys.path.append("./src")

import numpy as np
import pandas as pd
import utils.dists as udists
import utils.data as udata
import utils.misc as u
import losses
import os
from tqdm import tqdm
from glob import glob
from typing import List


data_dir = snakemake.input.data_dir
EXP_NAME = snakemake.config["EXP_NAME"]
exp_dir = os.path.join(data_dir, "processed", EXP_NAME)
TEST_SPLIT_THRESH = snakemake.config["TEST_SPLIT_THRESH"][EXP_NAME]

COMPONENT_NAMES = u.available_models(exp_dir)
COMPONENTS = udata.get_components(exp_dir, COMPONENT_NAMES)
ACTUAL_DL = udata.ActualDataLoader(data_dir)

REGIONS = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]
TARGET_NAMES = [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]


class Target:
    """
    Class collecting properties of a target
    """

    def __init__(self, name) -> None:
        self._name = name

    @property
    def name(self):
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
            return udists.BINS["wili"]
        else:
            return udists.BINS[self._name]

    @property
    def getter_fn(self):
        if self.type == "weekly":
            return udata.get_week_ahead_training_data

        else:
            return udata.get_seasonal_training_data

    def get_testing_data(self):
        """
        Return testing y, Xs, yi for target and all regions
        """

        y, Xs, yi = self.getter_fn(
            self._name, None,
            ACTUAL_DL, [c.loader for c in COMPONENTS]
        )

        train_indices = yi[:, 0] < TEST_SPLIT_THRESH

        return y[~train_indices], [X[~train_indices] for X in Xs], yi[~train_indices]


class Model:
    """
    Class collecting model properties
    """

    def __init__(self, name: str, weights_df: pd.DataFrame) -> None:
        self.name = name
        self.ws = weights_df

    def get_weights(self, target: Target, region: str) -> np.ndarray:
        """
        Return weights for component models in the order of COMPONENTS.
        """

        output = np.zeros(len(COMPONENTS))

        for i, c in enumerate(COMPONENTS):
            c_subset = self.ws[self.ws["model"] == c.name]

            if "target_type" in self.ws.columns:
                # This is target type weighing
                weight = c_subset[c_subset["target_type"] == target.type].iloc[0, :]["weight"]
            elif "region" in self.ws.columns:
                # This is target, region weighing
                weight = c_subset[
                    (c_subset["target"] == target.name) &
                    (c_subset["region"] == (region if region is not None else "all"))
                ].iloc[0, :]["weight"]
            elif "target" in self.ws.columns:
                # This is target weighing
                weight = c_subset[c_subset["target"] == target.name].iloc[0, :]["weight"]
            else:
                # This is constant/equal weighing
                weight = c_subset.iloc[0, :]["weight"]

            output[i] = weight
        return output


def dem_models(weight_files: str) -> List[Model]:
    """
    Return models from given weight directory
    """

    names = [os.path.basename(f).split(".")[0] for f in weight_files]
    dfs = [pd.read_csv(f) for f in weight_files]

    return [Model(n, w) for n, w in zip(names, dfs)]


# Entry point
for target in tqdm([Target(t) for t in TARGET_NAMES]):
    y, Xs, yi = target.get_testing_data()

    for model in dem_models(snakemake.input.w_files):
        eval_df = {
            "region": [],
            "score": []
        }
        output_dir = u.ensure_dir(f"./results/{EXP_NAME}/{target.name}")
        for region in REGIONS:
            if region is not None:
                region_indices = yi[:, 1] == region
                y_sub = y[region_indices]
                Xs_sub = [X[region_indices] for X in Xs]
                yi_sub = yi[region_indices]
            else:
                y_sub = y
                Xs_sub = Xs
                yi_sub =yi

            y_one_hot = udists.actual_to_one_hot(y_sub, bins=target.bins)

            weights = model.get_weights(target, region)
            output = udists.weighted_ensemble(Xs_sub, weights)

            eval_df["region"].append(region if region is not None else "all")
            eval_df["score"].append(losses.mean_cat_cross(y_one_hot, output))

        pd.DataFrame(eval_df).to_csv(f"{output_dir}/{model.name}.csv", index=False)
