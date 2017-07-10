
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

MODELS = ["kde", "kcde", "sarima"]

rule separate_model_data:
    input:
        ensemble_csv = "data/external/ensemble-data.csv",
        out_dir = "data/processed/"
    output:
        index = expand("data/processed/{model}/index.csv", model=MODELS),
        scores = expand("data/processed/{model}/scores.np.gz", model=MODELS),
        onset = expand("data/processed/{model}/onset.np.gz", model=MODELS),
        peak_wk = expand("data/processed/{model}/peak_wk.np.gz", model=MODELS),
        peak = expand("data/processed/{model}/peak.np.gz", model=MODELS),
        one_wk = expand("data/processed/{model}/one_wk.np.gz", model=MODELS),
        two_wks = expand("data/processed/{model}/two_wk.np.gz", model=MODELS),
        three_wks = expand("data/processed/{model}/three_wk.np.gz", model=MODELS),
        gour_wks = expand("data/processed/{model}/four_wk.np.gz", model=MODELS)
    script:
        "scripts/separate-model-data.py"
