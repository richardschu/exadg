#!/bin/bash
# Runs all runs of a study folder: writes the input files from its study.json,
# then runs each input with its own output folder and log file in the study
# folder. Usage:
#   ./run_study.sh <study folder> <path to solver> [runs at a time] [processes each]
here=$(dirname "$(realpath "$0")")
folder=$(realpath "$1")
solver=$(realpath "$2")
n_jobs=${3:-8}
n_procs=${4:-1}
python3 "$here/generate_inputs.py" "$folder" || exit 1
cd "$folder" || exit 1
export solver n_procs
ls input_*.json | xargs -P "$n_jobs" -I {} bash -c '
  name=$(basename {} .json); name=${name#input_}
  rm -rf output_${name}
  mpirun -n $n_procs --bind-to none "$solver" {} > log_${name}.txt 2>&1
  echo "$name: exit $?"'
