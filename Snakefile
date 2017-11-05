# Snakefile for processes

MODELS = ["kde", "kcde", "sarima"]

COMPONENTS_PATH = "data/processed/components/"

# Collect actual data from delphi API
# See the script to check if the data collected are from the current
# issue or are from the corresponding past week
rule get_actual_data:
    output:
        actual_csv = "data/processed/actual.csv",
        baseline_csv = "data/processed/baseline.csv"
    script: "scripts/get-actual-data.py"

# Clear component model data
rule clear_component_data:
    input: COMPONENTS_PATH
    shell: "cd {input}; rm -r *;"

# Use ensemble-data.csv to create separate prediction files for each model
rule separate_ensemble_data:
    input:
        ensemble_csv = "data/external/ensemble-data.csv.gz",
        out_dir = COMPONENTS_PATH
    output:
        index = expand(COMPONENTS_PATH + "{model}/index.csv", model=MODELS),
        scores = expand(COMPONENTS_PATH + "{model}/scores.np.gz", model=MODELS),
        onset = expand(COMPONENTS_PATH + "{model}/onset_wk.np.gz", model=MODELS),
        peak_wk = expand(COMPONENTS_PATH + "{model}/peak_wk.np.gz", model=MODELS),
        peak = expand(COMPONENTS_PATH + "{model}/peak.np.gz", model=MODELS),
        wk_1 = expand(COMPONENTS_PATH + "{model}/1.np.gz", model=MODELS),
        wk_2 = expand(COMPONENTS_PATH + "{model}/2.np.gz", model=MODELS),
        wk_3 = expand(COMPONENTS_PATH + "{model}/3.np.gz", model=MODELS),
        wk_4 = expand(COMPONENTS_PATH + "{model}/4.np.gz", model=MODELS)
    script: "scripts/separate-ensemble-data.py"

# Convert flusight style data directory to component data for this repository
rule pull_flusight_data:
    input:
        flusight_dir = "data/external/flusight-data-dir",
        out_dir = COMPONENTS_PATH
    script: "scripts/pull-flusight-data.py"

# Convert adaptively-weighted-ensemble data files to single ensemble-data.csv.gz
rule assemble_ensemble_data:
    input: "data/external/adaptively-weighted-ensemble/"
    output: "data/external/ensemble-data.csv.gz"
    script: "scripts/assemble-ensemble-data.R"
