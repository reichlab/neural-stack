"""
Generate submission files in flusight data directory
"""

import json
import sys
from glob import glob
from pathlib import Path

from tqdm import tqdm

sys.path.append("./src")
import predict
import submission


# Generate for current season only
flusight_data_dir = Path(snakemake.input[1]).joinpath("2016-2017")
models_dir = Path(snakemake.input[0])

targets = ["one-week", "two-weeks", "three-weeks", "four-weeks"]

model_files = [
    str(models_dir.joinpath(target, "cnn-1d-emb")) for target in targets
]

smoothing_files = [
    models_dir.joinpath(target, "cnn-1d-emb-smoothing-params") for target in targets
]

smoothing_params = []
for smoothing_file in smoothing_files:
    with smoothing_file.open() as f:
        smoothing_params.append(json.load(f))

models = [predict.load_model(m) for m in model_files]

component_models = ["KCDE", "KDE", "SARIMA"]

times = [int(f.name.split(".")[0]) for f in flusight_data_dir.joinpath(component_models[0]).glob("*.csv")]

for time in tqdm(times):
    sub_files = [str(flusight_data_dir.joinpath(cm, f"{time}.csv")) for cm in component_models]
    subs = [submission.read_csv(s, time, "", "") for s in sub_files]
    predict.generate_submission(models, subs, flusight_data_dir.joinpath("CNN", f"{time}.csv"), smoothing_params)
