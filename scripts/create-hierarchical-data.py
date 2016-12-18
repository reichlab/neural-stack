"""
Create h5 file with common rows across all
(year, week, region) triplets for models
"""

import pandas as pd
import numpy as np

# Read csv
df = pd.read_csv(snakemake.input[0])
df = df.drop(df.columns[[0]], axis=1)
df = df.dropna()

# Temporary key for getting common rows
key = (df["analysis_time_season"]
       + df["analysis_time_season_week"].map(str)
       + df["region"])
df["key"] = key

models = ["kde", "trunc_kde", "kcde", "sarima"]

intersection = np.intersect1d(df[df["model"] == models[0]]["key"],
                              df[df["model"] == models[1]]["key"])
for model in models[2:]:
    intersection = np.intersect1d(intersection,
                                  df[df["model"] == model]["key"])

df = df[df["key"].isin(intersection)]
df = df.drop("key", axis=1)

# Keys to export
values = ["onset_log_score",
          "peak_week_log_score",
          "peak_inc_log_score",
          "ph_1_inc_log_score",
          "ph_2_inc_log_score",
          "ph_3_inc_log_score",
          "ph_4_inc_log_score"]

indices = ["analysis_time_season", "analysis_time_season_week", "region"]

columns = ["model"]

df = df.pivot_table(index=indices,
                    columns=columns,
                    values=values)
df.to_hdf(snakemake.output[0], "data")
