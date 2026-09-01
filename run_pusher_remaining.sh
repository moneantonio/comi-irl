#!/bin/bash

# Pusher-v4 seeds 3 to 4
echo "=== Running Pusher-v4 seeds 3 to 4 ==="
for seed in {3..4}
do
  python deep_dp_birl_stress_test.py --env Pusher-v4 --seed $seed --iters 10 --local-new-component-steps 0 --verbose --mode bnirl_subgoal --dp-alpha 0.05
done
