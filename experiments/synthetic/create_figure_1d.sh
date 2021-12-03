#!/bin/bash

for num in {1..50};
do
    for c in 10 20 30 40 50 60 70 80 90 100; do
        python3 new_exp.py --exp_num=$num --loss=simloss --lower_bound=0.9 --c=$c --experiment=loss_vs_c
    done
done