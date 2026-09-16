#!/bin/bash

#SBATCH --job-name=pbmc10k
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate bmi500

cd /home/hguan28/bmi500

/usr/bin/time -v python -u scanpy_pbmc_profiling.py --data-dir data --data-set pbmc10k --out-dir data/outputs/ > pbmc10k_output.txt 2>&1