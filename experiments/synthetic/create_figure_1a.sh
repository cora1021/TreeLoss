#!/bin/bash

for num in {1..50};
do
    for n in 16 32 64 128 256 512 1024 2048 4096 8192; do
        python3 run_experiment.py --exp_num=$num --loss=HSM --n=$n --experiment=loss_vs_n
    done
done
