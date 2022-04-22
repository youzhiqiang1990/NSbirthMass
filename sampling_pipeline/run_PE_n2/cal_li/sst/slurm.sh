#!/bin/bash
#SBATCH --job-name=hyper
#SBATCH -p zhu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate pbenv

# The directory logs/ must exist before you submit this file

# Note npool must match the ntasks-per-node given above

python hyper.py
