#!/bin/bash

# for num in {1..50};
# do
for d_ in {601..800}; do
    python3 proj_exp.py --num=4 --k=1000 --loss=tree --d_=$d_ --experiment=loss_vs_d_
done
# done