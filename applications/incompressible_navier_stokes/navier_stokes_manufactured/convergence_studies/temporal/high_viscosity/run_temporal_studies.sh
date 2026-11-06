#!/bin/bash
# Runs all temporal convergence studies `input_temporal_*.json` in this folder, each
# with its own output folder and log file. Usage, from this folder:
#   ./run_temporal_studies.sh <path to solver> [studies at a time] [processes each]
solver=$(realpath "$1")
n_jobs=${2:-8}
n_procs=${3:-1}
export solver n_procs
ls input_temporal_*.json | xargs -P "$n_jobs" -I {} bash -c '
  name=$(basename {} .json); name=${name#input_}
  rm -rf output_${name}
  mpirun -n $n_procs --bind-to none "$solver" {} > log_${name}.txt 2>&1
  echo "$name: exit $?"'
