"""
Generate weights using DEM models
"""

import sys
sys.path.append("./src")

import numpy as np
import pandas as pd
import utils.dists as udists
import utils.data as udata
import utils.misc as u
import models
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

    def get_training_data(self, region=None):
        """
        Return training y, Xs, yi for target and all regions
        """

        y, Xs, yi = self.getter_fn(
            self._name, region,
            ACTUAL_DL, [c.loader for c in COMPONENTS]
        )

        train_indices = yi[:, 0] < TEST_SPLIT_THRESH

        return y[train_indices], [X[train_indices] for X in Xs], yi[train_indices]


def generate_equal_weights(output_file: str):
    """
    Generate equal weights df
    """

    weights = pd.DataFrame({
        "model": [c.name for c in COMPONENTS],
        "weight": [1/len(COMPONENTS) for c in COMPONENTS]
    })
    weights.to_csv(output_file, index=False)


def generate_constant_weights(output_file: str):
    """
    Generate constant weights using degenerate em
    """

    targets = [Target(name) for name in TARGET_NAMES]
    scores = []
    for target in targets:
        y, Xs, yi = target.get_training_data()
        scores.append(udists.score_predictions(Xs, y))

    scores = np.concatenate(scores, axis=0)
    weights = pd.DataFrame({
        "model": [c.name for c in COMPONENTS],
        "weight": models.dem(np.exp(scores))
    })
    weights.to_csv(output_file, index=False)


def generate_target_weights(output_file: str):
    """
    Generate weights based on targets using degenerate em
    """

    targets = [Target(name) for name in TARGET_NAMES]
    weights = {
        "model": [],
        "target": [],
        "weight": []
    }

    for target in targets:
        y, Xs, yi = target.get_training_data()
        scores = udists.score_predictions(Xs, y)

        weights["model"] += [c.name for c in COMPONENTS]
        weights["weight"] += list(models.dem(np.exp(scores)))
        weights["target"] += [target.name for c in COMPONENTS]

    pd.DataFrame(weights).to_csv(output_file, index=False)


def generate_target_type_weights(output_file: str):
    """
    Generate weights based on target types using degenerate em
    """

    targets = [Target(name) for name in TARGET_NAMES]
    weights = {
        "model": [],
        "target_type": [],
        "weight": []
    }

    def _append_target_type_weight(target_type):
        scores = []
        for target in targets:
            if target.type == target_type:
                y, Xs, yi = target.get_training_data()
                scores = udists.score_predictions(Xs, y)

        weights["model"] += [c.name for c in COMPONENTS]
        weights["weight"] += list(models.dem(np.exp(scores)))
        weights["target_type"] += [target_type for c in COMPONENTS]

    for target_type in ["weekly", "seasonal"]:
        _append_target_type_weight(target_type)

    pd.DataFrame(weights).to_csv(output_file, index=False)


def generate_target_region_weights(output_file: str):
    """
    Generate weights based on target and region using degenerate em
    """

    targets = [Target(name) for name in TARGET_NAMES]
    weights = {
        "model": [],
        "target": [],
        "region": [],
        "weight": []
    }

    regions = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]

    for target in targets:
        for region in regions:
            y, Xs, yi = target.get_training_data(region)
            scores = udists.score_predictions(Xs, y)
            weights["model"] += [c.name for c in COMPONENTS]
            weights["weight"] += list(models.dem(np.exp(scores)))
            weights["target"] += [target.name for c in COMPONENTS]
            weights["region"] += [(region if region is not None else "all") for c in COMPONENTS]

    pd.DataFrame(weights).to_csv(output_file, index=False)


# Generate weights
generate_equal_weights(f"./weights/{EXP_NAME}/dem-equal.csv")
generate_constant_weights(f"./weights/{EXP_NAME}/dem-constant.csv")
generate_target_weights(f"./weights/{EXP_NAME}/dem-target.csv")
generate_target_type_weights(f"./weights/{EXP_NAME}/dem-target-type.csv")
generate_target_region_weights(f"./weights/{EXP_NAME}/dem-target-region.csv")
