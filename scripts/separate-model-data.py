"""
Create model data files from ensemble-data.csv
"""

import gzip
import numpy as np
import os
import pandas as pd
import pymmwr


# Read predictions
with gzip.open(snakemake.input.ensemble_csv) as fp:
    df = pd.read_csv(fp, index_col=0)
df = df.dropna()

def get_epiweek(season: str, season_wk: int):
    """
    Get epiweek integer
    """

    season_first_year = int(season.split("/")[0])
    season_second_year = int(season.split("/")[1])

    # Week 31 is season_wk 01
    week = 30 + season_wk
    first_year_weeks = pymmwr.mmwr_weeks_in_year(season_first_year)

    epiweek_year = season_first_year
    if week > first_year_weeks:
        week -= first_year_weeks
        epiweek_year = season_second_year

    return int(str(epiweek_year) + str(week).zfill(2))

df["epiweek"] = list(map(get_epiweek, df["analysis_time_season"], df["analysis_time_season_week"]))
df.drop(["analysis_time_season", "analysis_time_season_week"], axis=1, inplace=True)

# Use short region code
df["region"] = df["region"].map({
    "National": "nat",
    "Region1": "hhs1",
    "Region2": "hhs2",
    "Region3": "hhs3",
    "Region4": "hhs4",
    "Region5": "hhs5",
    "Region6": "hhs6",
    "Region7": "hhs7",
    "Region8": "hhs8",
    "Region9": "hhs9",
    "Region10": "hhs10",
})

models = list(df["model"].unique())
for model in models:
    model_dir = os.path.join(snakemake.input.out_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    df_sub = df[df["model"] == model]
    identifiers = df_sub[["epiweek", "region"]]
    identifiers.to_csv(os.path.join(model_dir, "index.csv"), index=False)

    # Write log scores
    # First 7 are
    # 1. onset
    # 2. peak week
    # 3. peak incidence
    # 4. week 1
    # 5. week 2
    # 6. week 3
    # 7. week 4
    # Next 7 are competition log scores for the same targets as above
    np.savetxt(os.path.join(model_dir, "scores.np.gz"), df_sub.iloc[:, 0:14].values)

    # Write onset bin values
    # Bins go from season week 10 (epiweek 40) to season week 42 (epiweek
    # depends on the number of weeks in first year) and a none bin
    np.savetxt(os.path.join(model_dir, "onset.np.gz"), df_sub.iloc[:, 14:48].values)

    # Write peak week bin values
    # Bins go from season week 10 (epiweek 40) to season week 42
    np.savetxt(os.path.join(model_dir, "peak_wk.np.gz"), df_sub.iloc[:, 48:81].values)

    # Write peak incidence bin values
    # Bins from 0.0 to 13.0 in steps of 0.1
    np.savetxt(os.path.join(model_dir, "peak.np.gz"), df_sub.iloc[:, 81:212].values)

    # Write week ahead prediction bins
    start = 212
    for idx, week in enumerate(["one", "two", "three", "four"]):
        np.savetxt(os.path.join(model_dir, f"{week}_wk.np.gz"), df_sub.iloc[
            :, start + (idx * 131): start + ((idx + 1) * 131)
        ].values)
