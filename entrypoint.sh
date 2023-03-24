#!/bin/bash

# . /opt/conda/etc/profile.d/conda.sh
conda activate kws-env
uvicorn main:app --workers 1 --host 0.0.0.0 --port 8000