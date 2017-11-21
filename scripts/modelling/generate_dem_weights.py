"""
Generate weights using DEM models
"""

import sys
sys.path.append("./src")

import numpy as np
import pandas as pd
import utils.dists as udists
import utils.data as udata
import utils.misc as u
import models
import os
from tqdm import tqdm
from glob import glob
from typing import List


data_dir = snakemake.input.data_dir
EXP_NAME = snakemake.config["EXP_NAME"]
TEST_SPLIT_THRESH = snakemake.config["TEST_SPLIT_THRESH"][EXP_NAME]

COMPONENT_NAMES = u.available_models(data_dir)
COMPONENTS = udata.get_components(data_dir, COMPONENT_NAMES)
ACTUAL_DL = udata.ActualDataLoader(data_dir)

REGIONS = ["nat", *[f"hhs{i}" for i in range(1, 11)], None]
TARGET_NAMES = [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]
