#!/bin/bash

for num in {1..50};
do
    for d in 2 4 8 16 32 64 128 256 512 1024; do
        python3 run_experiment.py --exp_num=$num --loss=xentropy --d=$d --experiment=loss_vs_d 
    done
done