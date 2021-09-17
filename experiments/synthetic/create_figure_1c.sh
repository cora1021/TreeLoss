#!/bin/bash

for num in {1..50};
do
    for sigma in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do
        python3 run_experiment.py --exp_num=$num --sigma=$sigma --loss=simloss --lower_bound=0.6 --experiment=loss_vs_sigma
    done
done