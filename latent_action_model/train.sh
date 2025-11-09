torchrun --standalone --nnodes 1 --nproc-per-node 7 main.py fit \
    --config config/lam.yaml \
    2>&1 | tee lam.log
