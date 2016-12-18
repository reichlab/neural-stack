from os.path import join

configfile: "config.yaml"

rule all:
    input:
        "data/processed/stacking-data.h5"

rule create_json:
    input:
        "data/external/stacking-data.csv"
    output:
        "data/processed/stacking-data.h5"

    script:
        "scripts/create-hierarchical-data.py"

rule assemble_ensemble_data:
    input:
        "data/external/contest-repo"
    output:
        "data/external/stacking-data.csv"

    script:
        "scripts/collect-stacking-data.R"

rule get_contest_repo:
    output:
        "data/external/contest-repo"

    shell:
        "git clone --depth 1 " + config["contest-repo"] + " {output}"
