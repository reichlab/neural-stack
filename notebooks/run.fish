#!/usr/bin/env fish
## Notebook runner

function run_notebook
  set -l script (echo $argv[1] | sed "s/\.ipynb/.py/")

  rm -f $script
  jupyter nbconvert --to script $argv[1]

  sed -i "/get_ipython()/d" $script

  # Assuming target is set as 0
  sed -i "s/target = TARGETS\[0\]/target = TARGETS\[$argv[3]\]/" $script

  # Assuming exp_name is set as collaborative
  sed -i "s/EXP_NAME = \"collaborative\"/EXP_NAME = \"$argv[2]\"/" $script

  # Run stuff
  pipenv run python $script

  # Start new
  rm -f $script
end

# Args are notebook, exp_name, target_index
for e in "collaborative" "lab"
  for i in (seq 0 6)
    echo "Running experiment $e for target number $i on file $argv[1]"
    run_notebook $argv[1] $e $i
  end
end
