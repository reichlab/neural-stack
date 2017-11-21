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
from typing import List


data_dir = snakemake.input.data_dir
EXP_NAME = snakemake.config["EXP_NAME"]
exp_dir = os.path.join(data_dir, "processed", EXP_NAME)
TEST_SPLIT_THRESH = snakemake.config["TEST_SPLIT_THRESH"][EXP_NAME]

COMPONENTS = [udata.Component(exp_dir, name) for name in u.available_models(exp_dir)]
ACTUAL_DL = udata.ActualDataLoader(data_dir)

REGIONS = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]
TARGETS = [udata.Target(t) for t in [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]]


class Model:
    """
    Class collecting model properties
    """

    def __init__(self, name: str, weights_df: pd.DataFrame) -> None:
        self.name = name
        self.ws = weights_df

    def get_weights(self, target: udata.Target, region: str) -> np.ndarray:
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
for target in tqdm(TARGETS):
    for model in dem_models(snakemake.input.w_files):
        eval_df = {
            "region": [],
            "score": []
        }
        output_dir = u.ensure_dir(f"./results/{EXP_NAME}/{target.name}")
        for region in REGIONS:
            y, Xs, yi = target.get_testing_data(ACTUAL_DL, COMPONENTS, region, TEST_SPLIT_THRESH)
            y_one_hot = udists.actual_to_one_hot(y, bins=target.bins)

            weights = model.get_weights(target, region)
            output = udists.weighted_ensemble(Xs, weights)

            eval_df["region"].append(region if region is not None else "all")
            eval_df["score"].append(losses.mean_cat_cross(y_one_hot, output))

        pd.DataFrame(eval_df).to_csv(f"{output_dir}/{model.name}.csv", index=False)
