#!/bin/bash

for n in 16 32 64 128 256 512 1024 2048 4096 8192; do
    python3 experiments/synthetic/run_experiment.py --n=$n --experiment=loss_vs_n
done
