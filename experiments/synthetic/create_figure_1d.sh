#!/bin/bash

for num in {1..50};
do
    for c in 10 20 30 40 50 60 70 80 90 100; do
        python3 run_experiment.py --exp_num=$num --loss=HSM --c=$c --experiment=loss_vs_c
    done
done