#!/bin/bash

for n in 2 4 8 16 32 64 128; do
    python3 experiments/synthetic/run_experiment.py --n=$n
done
