#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=8223

../env.sh
. activate
conda activate lhy_pytorch

python main.py

