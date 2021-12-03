#!/bin/bash

# for num in {1..50};
# do
for a in {5..200}; do
    python3 proj_exp.py --num=1 --loss=tree --a=$a --experiment=loss_vs_d_
done
# done