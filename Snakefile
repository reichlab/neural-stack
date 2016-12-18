from os.path import join

configfile: "config.yaml"

rule all:
    input:
        "data/external/stacking-data.csv"

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
