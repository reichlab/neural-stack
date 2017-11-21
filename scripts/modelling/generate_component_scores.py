"""
Generate scores for the components
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


# Entry point
for target in tqdm([Target(t) for t in TARGET_NAMES]):
    y, Xs, yi = target.get_testing_data()

    output_dir = u.ensure_dir(f"./results/{EXP_NAME}/{target.name}")

    for idx, cmp in enumerate(COMPONENTS):
        eval_df = {
            "region": [],
            "score": []
        }
        for region in REGIONS:
            if region is not None:
                region_indices = yi[:, 1] == region
                y_sub = y[region_indices]
                X_sub = Xs[idx][region_indices]
                yi_sub = yi[region_indices]

            y_one_hot = udists.actual_to_one_hot(y_sub, bins=target.bins)
            output = X_sub

            eval_df["region"].append(region if region is not None else "all")
            eval_df["score"].append(losses.mean_cat_cross(y_one_hot, output))

        pd.DataFrame(eval_df).to_csv(f"{output_dir}/{cmp.name}.csv", index=False)
