"""
Create h5 file for python models
"""

import numpy as np
import pandas as pd

# Read actual data
actual = pd.read_csv(snakemake.input[1])

def region_map(region):
    if region == "X":
        return "National"
    else:
        return region.replace(" ", "")

actual["region"] = actual["region"].apply(region_map)
actual["season"] = actual["season"].str.split("/").str[0]
actual["time"] = actual["season"].str.cat(actual["season_week"].astype("str").str.zfill(2)).astype("int")

actual = actual.loc[:, ["region", "weighted_ili", "time"]]
actual.columns = ["region", "actual", "time"]


# Read predictions
df = pd.read_csv(snakemake.input[0], index_col=0)
df = df.dropna()

df["analysis_time_season"] = df["analysis_time_season"].str.split("/").str[0]
df["time"] = df["analysis_time_season"].str.cat(df["analysis_time_season_week"].astype("str")).astype("int")

df.drop(["analysis_time_season", "analysis_time_season_week"], axis=1, inplace=True)

identifiers = ["model", "region", "time"]
df = pd.melt(df, id_vars = identifiers)

# Write to file
with pd.HDFStore(snakemake.output[0]) as store:
    store["predictions"] = df
    store["actual"] = actual
