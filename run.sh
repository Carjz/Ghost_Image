#!/bin/bash

# load environment
pip install -r requirement.txt > /dev/null

# training mode
python main.py

# evaluation mode
# python eval.py

