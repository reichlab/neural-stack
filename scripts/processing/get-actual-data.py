"""
Collect actual wili data using the delphi API
"""

from delphi_epidata import Epidata
from datetime import datetime
import pandas as pd
import pymmwr

BASELINE_URL = "https://raw.githubusercontent.com/cdcepi/FluSight-forecasts/master/wILI_Baseline.csv"
current_epiweek = pymmwr.date_to_mmwr_week()

# Range of epiweeks to gather data for
epiweek_start = 199710
epiweek_end = int(str(current_epiweek["year"]) + str(current_epiweek["week"]).zfill(2))

epiweek_range = Epidata.range(epiweek_start, epiweek_end)

regions = ["nat", *["hhs" + str(i) for i in range(1, 11)]]

# NOTE Lag value
# A lag of 0 means that the data for each week collected will be
# as observed at that point in time.
# Pass None as lag will let us collect the most recent data
# available

df = {
    "epiweek": [],
    "region": [],
    "wili": []
}

for region in regions:
    res = Epidata.fluview(region, epiweek_range, lag=None)
    for data in res["epidata"]:
        df["epiweek"].append(data["epiweek"])
        df["region"].append(data["region"])
        df["wili"].append(data["wili"])

# Write to file
pd.DataFrame(df).to_csv(snakemake.output.actual_csv, index=False)

# Save baseline information
bdf = pd.read_csv(BASELINE_URL, index_col=0)

def rename_region(region: str) -> str:
    if region == "National":
        return "nat"
    else:
        return f"hhs{region[6:]}"

def rename_season(season: str) -> int:
    return int(season.split("/")[0])

bdf.index = bdf.index.map(rename_region)
bdf.columns = bdf.columns.map(rename_season)
seasons = list(bdf.columns)
bdf = bdf.reset_index().rename(columns={"index": "region"})
bdf = bdf.melt(id_vars=["region"], value_vars=seasons, var_name="season", value_name="baseline")

def change_season(subset, new_year):
    """
    Return a copy of subset with season changed
    """

    df = subset.copy()
    df["season"] = new_year
    return df

# Use baseline of 2007 for seasons before it
past_seasons = range(1990, 2007)
copies = [change_season(bdf[bdf["season"] == 2007], year) for year in past_seasons]
bdf = pd.concat([*copies, bdf], ignore_index=True)

bdf.to_csv(snakemake.output.baseline_csv, index=False)
