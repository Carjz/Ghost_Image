#!/bin/bash

# load environment
. activate
conda activate lhy_pytorch
pip install -r requirement.txt > /dev/null

# training mode
python main.py

# evaluation mode
# python eval.py

