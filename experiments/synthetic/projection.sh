
#!/bin/bash

# for num in {1..50};
# do
for d_ in {5..1000}; do
    python3 proj_exp.py --k=1000 --loss=xentropy --d_=$d_ --experiment=loss_vs_d_
done
# done