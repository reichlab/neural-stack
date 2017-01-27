from os.path import join

configfile: "config.yaml"

rule all:
    input:
        "data/processed/ensemble-data.h5",
        "data/processed/ensemble-data-with-actual.h5"

rule add_actual_data:
    input:
        "data/processed/ensemble-data.h5",
        config["ensemble-repo-local"] + "/data-raw/allflu-cleaned.csv"

    output:
        "data/processed/ensemble-data-with-actual.h5"

    script:
        "scripts/add-actual-data.py"

rule create_h5:
    input:
        "data/external/ensemble-data.csv"
    output:
        "data/processed/ensemble-data.h5"

    script:
        "scripts/create-hierarchical-data.py"

rule assemble_ensemble_data:
    input:
        config["ensemble-repo-local"]
    output:
        "data/external/ensemble-data.csv"

    script:
        "scripts/collect-ensemble-data.R"

rule get_ensemble_repo:
    output:
        config["ensemble-repo-local"]

    shell:
        "git clone --depth 1 " + config["ensemble-repo"] + " {output}"
