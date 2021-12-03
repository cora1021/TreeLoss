#!/bin/bash

# for num in {1..50};
# do
for d_ in {201..400}; do
    python3 proj_exp.py --num=2 --k=1000 --loss=tree --d_=$d_ --experiment=loss_vs_d_
done
# done