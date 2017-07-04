from os.path import join

configfile: "config.yaml"

rule all:
    input:
        "data/processed/ensemble-data.h5"

rule generate_submissions:
    input:
        "data/models",
        config["flusight-data-dir"]
    script:
        "scripts/create-submission-files.py"

rule create_h5:
    input:
        "data/external/ensemble-data.csv",
        config["ensemble-repo-local"] + "/data-raw/allflu-cleaned.csv"
    output:
        "data/processed/ensemble-data.h5"

    script:
        "scripts/create-hierarchical-data.py"
