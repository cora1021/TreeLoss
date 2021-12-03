#!/bin/bash

# for num in {1..50};
# do
for d_ in {401..600}; do
    python3 proj_exp.py --num=3 --k=1000 --loss=tree --d_=$d_ --experiment=loss_vs_d_
done
# done