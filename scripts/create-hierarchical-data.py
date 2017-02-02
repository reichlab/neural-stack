"""
Create h5 file for python models
"""

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
actual = actual[["region", "time", "actual"]]


# Read predictions
df = pd.read_csv(snakemake.input[0], index_col=0)
df = df.dropna()

df["analysis_time_season"] = df["analysis_time_season"].str.split("/").str[0]
df["time"] = df["analysis_time_season"].str.cat(df["analysis_time_season_week"].astype("str")).astype("int")
df.drop(["analysis_time_season", "analysis_time_season_week"], axis=1, inplace=True)

# Separate column groups for clarity
identifiers = df[["model", "region", "time"]]

groups = [
    "scores",
    "onset",
    "peak_week",
    "peak",
    "one_week",
    "two_weeks",
    "three_weeks",
    "four_weeks"
]

# Write to file
with pd.HDFStore(snakemake.output[0]) as store:
    store["predictions/identifiers"] = identifiers
    store["actual"] = actual

    group_lengths = [14, 34, 33, 131, 131, 131, 131, 131]
    start = end = 2
    for g, l in zip(groups, group_lengths):
        key = "predictions/" + g
        end = start + l
        store[key] = df.iloc[:, start:end]
        start = end
