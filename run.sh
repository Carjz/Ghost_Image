#!/bin/bash

#SBATCH -o slurm.out

rm std.log err.log
touch std.log err.log

# load environment
. activate
conda activate lhy_pytorch
pip install -r requirement.txt > /dev/null

# training mode
python main.py >> std.log 2>> err.log

# evaluation mode
# python eval.py

