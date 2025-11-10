#!/bin/bash

set -e
eval "$(/home/dengyixuan/mzh/miniconda3/bin/conda shell.bash hook)"
conda activate base
export LD_LIBRARY_PATH=/home/dengyixuan/mzh/miniconda3/lib:$LD_LIBRARY_PATH

cd /home/dengyixuan/mzh/Code/UniVLA/openpi
source .venv/bin/activate

cd latent_action_model

torchrun --standalone --nnodes 1 --nproc-per-node 7 main.py fit \
    --config config/lam.yaml \
    2>&1 | tee lam.log
