from os.path import join

configfile: "config.yaml"

rule all:
    input:
        "data/processed/stacking-data.h5",
        "data/processed/stacking-data-with-actual.h5"

rule add_actual_data:
    input:
        "data/processed/stacking-data.h5",
        config["contest-repo-local"] + "/data-raw/allflu-cleaned.csv"

    output:
        "data/processed/stacking-data-with-actual.h5"

    script:
        "scripts/append-actual-data.py"

rule create_h5:
    input:
        "data/external/stacking-data.csv"
    output:
        "data/processed/stacking-data.h5"

    script:
        "scripts/create-hierarchical-data.py"

rule assemble_ensemble_data:
    input:
        config["contest-repo-local"]
    output:
        "data/external/stacking-data.csv"

    script:
        "scripts/collect-stacking-data.R"

rule get_contest_repo:
    output:
        config["contest-repo-local"]

    shell:
        "git clone --depth 1 " + config["contest-repo"] + " {output}"
