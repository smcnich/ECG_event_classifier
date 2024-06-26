#!/bin/bash
#SBATCH --job-name=mcni_load1
#SBATCH --output=load1.out
#SBATCH --partition=gpu
#SBATCH --nodelist=nedc_008

# source the .bashrc file
#
. ~/.bashrc

# activate the venv with torch
#
conda activate tensor

# Run your Python script
python -u /home/tuo72868/ece_8527_final/src/dataload.py -o /home/tuo72868/ece_8527_final/data/stacked -n $1
