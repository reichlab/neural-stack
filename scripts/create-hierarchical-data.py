"""
Create h5 file with common rows across all
(year, week, region) triplets for models
"""

import numpy as np
import pandas as pd

# Read csv
df = pd.read_csv(snakemake.input[0], index_col=0)
df = df.dropna()

identifiers = ["model", "region", "analysis_time_season", "analysis_time_season_week"]
df = pd.melt(df, id_vars = identifiers)

df.to_hdf(snakemake.output[0], "data")
