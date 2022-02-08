#!/bin/bash

for num in {1..50};
do
    for base in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 4.0 5.0 6.0 7.0; do
        python3 base_exp.py --exp_num=$num --loss=xentropy --base=$base
    done
done
