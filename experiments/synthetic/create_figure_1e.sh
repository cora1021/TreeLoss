#!/bin/bash

for num in {1..50};
do
    for random in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        python3 run_experiment.py --exp_num=$num --random=$random --loss=tree --experiment=loss_vs_structure
    done
done
