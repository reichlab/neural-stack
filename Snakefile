MODELS = ["kde", "kcde", "sarima"]
COMPONENTS_PATH = "data/processed/components/"

# Collect actual data from delphi API
# See the script to check if the data collected are from the current
# issue or are from the corresponding past week
rule actual_data:
    output:
        "data/processed/actual.csv"
    script:
        "scripts/get-actual-data.py"

# Use ensemble-data.csv to create separate prediction files for each
# model (KDE, KCDE, SARIMA)
rule separate_model_data:
    input:
        ensemble_csv = "data/external/ensemble-data.csv.gz",
        out_dir = COMPONENTS_PATH
    output:
        index = expand(COMPONENTS_PATH + "{model}/index.csv", model=MODELS),
        scores = expand(COMPONENTS_PATH + "{model}/scores.np.gz", model=MODELS),
        onset = expand(COMPONENTS_PATH + "{model}/onset.np.gz", model=MODELS),
        peak_wk = expand(COMPONENTS_PATH + "{model}/peak_wk.np.gz", model=MODELS),
        peak = expand(COMPONENTS_PATH + "{model}/peak.np.gz", model=MODELS),
        one_wk = expand(COMPONENTS_PATH + "{model}/one_wk.np.gz", model=MODELS),
        two_wks = expand(COMPONENTS_PATH + "{model}/two_wk.np.gz", model=MODELS),
        three_wks = expand(COMPONENTS_PATH + "{model}/three_wk.np.gz", model=MODELS),
        four_wks = expand(COMPONENTS_PATH + "{model}/four_wk.np.gz", model=MODELS)
    script:
        "scripts/separate-model-data.py"
