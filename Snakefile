
# Collect actual data from delphi API
# See the script to check if the data collected are from the current
# issue or are from the corresponding past week
rule actual_data:
    output:
        "data/external/actual.csv"
    script:
        "scripts/get-actual-data.py"
