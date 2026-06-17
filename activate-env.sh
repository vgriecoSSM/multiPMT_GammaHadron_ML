#!/bin/bash

ENV_NAME="gamma-hadron-multiPMT-env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Activated Conda environment: $ENV_NAME"
