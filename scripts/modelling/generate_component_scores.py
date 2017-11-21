"""
Generate scores for the components
"""

import sys
sys.path.append("./src")

import pandas as pd
import utils.dists as udists
import utils.data as udata
import utils.misc as u
import losses
import os
from tqdm import tqdm


data_dir = snakemake.input.data_dir
EXP_NAME = snakemake.config["EXP_NAME"]
exp_dir = os.path.join(data_dir, "processed", EXP_NAME)
TEST_SPLIT_THRESH = snakemake.config["TEST_SPLIT_THRESH"][EXP_NAME]

COMPONENTS = [udata.Component(exp_dir, name) for name in u.available_models(exp_dir)]
ACTUAL_DL = udata.ActualDataLoader(data_dir)

REGIONS = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]
TARGETS = [udata.Target(t) for t in [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]]


# Entry point
for target in tqdm(TARGETS):
    output_dir = u.ensure_dir(f"./results/{EXP_NAME}/{target.name}")

    for idx, cmp in enumerate(COMPONENTS):
        eval_df = {
            "region": [],
            "score": []
        }
        for region in REGIONS:
            y, Xs, yi = target.get_testing_data(ACTUAL_DL, COMPONENTS, region, TEST_SPLIT_THRESH)
            y_one_hot = udists.actual_to_one_hot(y, bins=target.bins)
            output = Xs[idx]

            eval_df["region"].append(region if region is not None else "all")
            eval_df["score"].append(losses.mean_cat_cross(y_one_hot, output))

        pd.DataFrame(eval_df).to_csv(f"{output_dir}/{cmp.name}.csv", index=False)
