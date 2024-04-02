#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=8223

module load apps/anaconda3/
. activate
conda activate pytorch

python main.py

