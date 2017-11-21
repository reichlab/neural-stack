# Snakefile for processes

configfile: "config.yaml"

TARGET_NAMES = [1, 2, 3, 4, "peak", "peak_wk", "onset_wk"]

# Collect actual data from delphi API
# See the script to check if the data collected are from the current
# issue or are from the corresponding past week
rule get_actual_data:
    output:
        actual_csv = "data/processed/actual.csv",
        baseline_csv = "data/processed/baseline.csv"
    script: "scripts/processing/get-actual-data.py"


# Lab experiment setting
# ----------------------
rule process_lab_dir:
    input: "data/external/adaptively-weighted-ensemble/"
    output: "data/external/lab/"
    run:
        shell("mkdir -p {output}")
        shell("cp data/external/adaptively-weighted-ensemble/inst/evaluation/test-predictions/*.rds {output}")
        shell("cp data/external/adaptively-weighted-ensemble/inst/estimation/loso-predictions/*.rds {output}")

# Convert adaptively-weighted-ensemble data files to single ensemble-data.csv.gz
rule preprocess_lab_data:
    input: "data/external/lab/"
    output: "data/external/ensemble-data.csv.gz"
    script: "scripts/processing/preprocess-lab-data.R"

lab_models = ["kcde", "kde", "sarima"]

# Use ensemble-data.csv to create separate prediction files for each model
rule get_lab_data:
    input: "data/external/ensemble-data.csv.gz"
    output:
        out_dir = "data/processed/lab/",
        index = expand("data/processed/lab/{model}/index.csv", model=lab_models),
        scores = expand("data/processed/lab/{model}/scores.np.gz", model=lab_models),
        files = expand("data/processed/lab/{model}/{target}.np.gz", model=lab_models, target=TARGET_NAMES)
    script: "scripts/processing/get-lab-data.py"


# Collaborative experiment setting
# --------------------------------
rule process_collaborative_dir:
    input: "data/external/cdc-flusight-ensemble/"
    output: "data/external/collaborative/"
    run:
        shell("cd data/external/cdc-flusight-ensemble; yarn")
        shell("cd data/external/cdc-flusight-ensemble/flusight-deploy; yarn; yarn run parse-data")
        shell("mv data/external/cdc-flusight-ensemble/flusight-deploy/data/* data/external/collaborative/")
        shell("rm -rf data/external/collaborative/2017-2018")

# Convert flusight style data directory to component data for this repository
rule get_collaborative_data:
    input: "data/external/collaborative/"
    output: "data/processed/collaborative/"
    script: "scripts/processing/get-collaborative-data.py"


# Component model scoring
# -----------------------
# rule generate_component_scores:


# Other misc ensemble scoring
# ---------------------------
# rule generate_product_scores:


# Degenerate EM
# -------------
dem_models = [
    "dem-constant",
    "dem-equal",
    "dem-target",
    "dem-target-region",
    "dem-target-type"
]

# Generate weights using dem
rule generate_dem_weights:
    input:
        data_dir = "data/",
        w_dir = "weights/"
    output: expand("weights/{exp_name}/{model}.csv", exp_name=config["EXP_NAME"], model=dem_models)
    script: "scripts/modelling/generate_dem_weights.py"

# Generate scores for dem models using weights
rule generate_dem_scores:
    input: expand("weights/{exp_name}/{model}.csv", exp_name=config["EXP_NAME"], model=dem_models)
    output: expand("results/{exp_name}/{target}/{model}.csv", exp_name=config["EXP_NAME"], target=TARGET_NAMES, model=dem_models)
    script: "scripts/modelling/generate_dem_scores.py"
