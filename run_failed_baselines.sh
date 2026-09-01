#!/bin/bash

# Reacher-v4 seeds 0 to 4
echo "=== Running Reacher-v4 seeds 0 to 4 ==="
for seed in {0..4}
do
  python deep_dp_birl_stress_test.py --env Reacher-v4 --seed $seed --iters 10 --local-new-component-steps 0 --verbose --mode bnirl_subgoal --dp-alpha 0.05
done

# Pusher-v4 seeds 0 to 2
echo "=== Running Pusher-v4 seeds 0 to 2 ==="
for seed in {0..2}
do
  python deep_dp_birl_stress_test.py --env Pusher-v4 --seed $seed --iters 10 --local-new-component-steps 0 --verbose --mode bnirl_subgoal --dp-alpha 0.05
done
