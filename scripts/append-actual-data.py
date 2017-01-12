"""
Add actual data to the h5 file
"""

import pandas as pd
from pathlib import Path

df = pd.read_hdf(snakemake.input[0], "data")
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

# Pivot to multiIndex
values = ["actual"]
indices = ["analysis_time_season", "analysis_time_season_week", "region"]
actual = actual.pivot_table(index=indices, values=values)
merged = actual.join(df, how="inner")

merged.to_hdf(snakemake.output[0], "data")
