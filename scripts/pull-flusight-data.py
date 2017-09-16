"""
Pull in flusight data to this repo component models
"""


import gzip
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

sys.path.append("./src")
import submission


def parse_csv(csv_path: Path):
    """
    Parse csv file to return rows for storing
    """

    epiweek = int(csv_path.stem)
    sub = submission.Submission(pd.read_csv(str(csv_path)))

    n_regions = len(submission.MAP_REGION)

    index = {
        "epiweek": [epiweek] * n_regions,
        "region": []
    }

    matrices = {
        1: np.zeros((n_regions, 131)),
        2: np.zeros((n_regions, 131)),
        3: np.zeros((n_regions, 131)),
        4: np.zeros((n_regions, 131)),
        "onset_wk": np.zeros((n_regions, 34)),
        "peak_wk": np.zeros((n_regions, 33)),
        "peak": np.zeros((n_regions, 131))
    }

    for idx, item in enumerate(submission.MAP_REGION.items()):
        index["region"].append(item[0])
        for target in matrices:
            # NOTE: The data from ensemble loader uses log values instead of normal
            # Here we are using normal values directly
            try:
                matrices[target][idx, :] = sub.get_X(item[0], target)
            except ValueError:
                print(f"shape {sub.get_X(item[0], target).shape} {target} for region {item[0]} file name : {csv_path}")
    return [pd.DataFrame(index), matrices]

def get_csvs(directory: Path):
    """
    Return list of csv files from the directory
    """

    return [item for item in directory.iterdir() if item.is_file() and item.suffix == '.csv']

def get_model_csvs(data_dir: Path):
    """
    Return a map of model name and csvs for it
    """

    season_dirs = [item for item in data_dir.iterdir() if item.is_dir()]

    output = {}

    for season_dir in season_dirs:
        model_dirs = [item for item in season_dir.iterdir() if item.is_dir()]
        for model_dir in model_dirs:
            model_name = model_dir.stem
            if model_name not in output:
                output[model_name] = get_csvs(model_dir)
            else:
                output[model_name] += get_csvs(model_dir)

    return output

def write_model_data(output_path, index, matrices):
    """
    Write model data to given path
    """

    os.makedirs(output_path)
    index.to_csv(output_path.joinpath("index.csv"), index=False)

    for target in matrices:
        np.savetxt(output_path.joinpath(f"{target}.np.gz"), matrices[target])


def merge_indices(indices):
    """
    Merge list of indices into a single index
    """

    return pd.concat(indices)

def merge_matrices(matrices):
    """
    Merge list of matrices into single one
    """

    # Get number of regions from 1 week ahead matrix for first csv
    n_regions = matrices[0][1].shape[0]
    n_rows = n_regions * len(matrices) # len(matrices) is no. of csvs
    output = {
        1: None,
        2: None,
        3: None,
        4: None,
        "onset_wk": None,
        "peak_wk": None,
        "peak": None
    }
    for target in output:
        output[target] = np.concatenate([mat[target] for mat in matrices], axis=0)
    return output

# E N T R Y  P O I N T
# --------------------
# Walk through the data directory
model_csvs = get_model_csvs(Path(snakemake.input.flusight_dir))

for model in model_csvs:
    print(f"Parsing {model}")
    parsed = [parse_csv(csv) for csv in tqdm(model_csvs[model])]
    all_indices = [p[0] for p in parsed]
    all_matrices = [p[1] for p in parsed]

    merged_index = merge_indices(all_indices)
    merged_matrices = merge_matrices(all_matrices)

    # Write to directory
    model_output_dir = Path(snakemake.input.output_dir).joinpath(model)
    write_model_data(model_output_dir, merged_index, merged_matrices)
