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
from typing import List


data_dir = snakemake.input.data_dir
EXP_NAME = snakemake.config["EXP_NAME"]
exp_dir = os.path.join(data_dir, "processed", EXP_NAME)
TEST_SPLIT_THRESH = snakemake.config["TEST_SPLIT_THRESH"][EXP_NAME]

COMPONENTS = [udata.Component(exp_dir, name) for name in u.available_models(exp_dir)]
ACTUAL_DL = udata.ActualDataLoader(data_dir)

REGIONS = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]
TARGETS = [udata.Target(t) for t in [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]]


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

    scores = []
    for target in TARGETS:
        y, Xs, yi = target.get_training_data(ACTUAL_DL, COMPONENTS, None, TEST_SPLIT_THRESH)
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

    weights = {
        "model": [],
        "target": [],
        "weight": []
    }

    for target in TARGETS:
        y, Xs, yi = target.get_training_data(ACTUAL_DL, COMPONENTS, None, TEST_SPLIT_THRESH)
        scores = udists.score_predictions(Xs, y)

        weights["model"] += [c.name for c in COMPONENTS]
        weights["weight"] += list(models.dem(np.exp(scores)))
        weights["target"] += [target.name for c in COMPONENTS]

    pd.DataFrame(weights).to_csv(output_file, index=False)


def generate_target_type_weights(output_file: str):
    """
    Generate weights based on target types using degenerate em
    """

    weights = {
        "model": [],
        "target_type": [],
        "weight": []
    }

    def _append_target_type_weight(target_type):
        scores = []
        for target in TARGETS:
            if target.type == target_type:
                y, Xs, yi = target.get_training_data(ACTUAL_DL, COMPONENTS, None, TEST_SPLIT_THRESH)
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

    weights = {
        "model": [],
        "target": [],
        "region": [],
        "weight": []
    }

    for target in TARGETS:
        for region in REGIONS:
            y, Xs, yi = target.get_training_data(ACTUAL_DL, COMPONENTS, region, TEST_SPLIT_THRESH)
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
