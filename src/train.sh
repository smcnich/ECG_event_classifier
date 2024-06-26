#!/bin/bash
#SBATCH --job-name=rf_mcni
#SBATCH --output=rf.out
#SBATCH --nodelist=nedc_012
#SBATCH --partition=gpu

# source the .bashrc file
#
. ~/.bashrc

# activate the venv with torch
#
conda activate tensor

# Run your Python script
python -u /home/tuo72868/ece_8527_final/src/train.py -o ../model/ -m RF /home/tuo72868/ece_8527_final/data/feats/train.pkl