for sigma in 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25; do
    python3 experiments/synthetic/run_experiment.py --sigma=$sigma --experiment=loss_vs_sigma --max_iter=1
done