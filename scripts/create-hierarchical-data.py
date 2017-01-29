"""
Create h5 file for python models
"""

import numpy as np
import pandas as pd

# Read csv
df = pd.read_csv(snakemake.input[0], index_col=0)
df = df.dropna()

actual = pd.read_csv(snakemake.input[1])

def region_map(region):
    if region == "X":
        return "National"
    else:
        return region.replace(" ", "")

actual = actual.loc[:, ["region", "weighted_ili", "season", "season_week"]]
actual["region"] = actual["region"].apply(region_map)
actual.columns = ["region", "actual",
                  "analysis_time_season", "analysis_time_season_week"]

identifiers = ["model", "region", "analysis_time_season", "analysis_time_season_week"]
df = pd.melt(df, id_vars = identifiers)

# Write to file
with pd.HDFStore(snakemake.output[0]) as store:
    store["predictions"] = df
    store["actual"] = actual
