"""
Collect actual wili data using the delphi API
"""

from delphi_epidata import Epidata
from datetime import datetime
import pandas as pd
import pymmwr


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
pd.DataFrame(df).to_csv(snakemake.output[0], index=False)
