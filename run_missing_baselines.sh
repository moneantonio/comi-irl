#!/bin/bash

# Array of environments
ENVIRONMENTS=("Reacher-v4" "Pusher-v4")

# Array of seeds
SEEDS=(0 1 2 3 4)

for env in "${ENVIRONMENTS[@]}"
do
  for seed in "${SEEDS[@]}"
  do
    echo "========================================================================="
    echo "Rerunning BNIRL Subgoal on environment $env with seed $seed"
    echo "========================================================================="
    
    python deep_dp_birl_stress_test.py \
      --env "$env" \
      --seed "$seed" \
      --iters 10 \
      --local-new-component-steps 0 \
      --verbose \
      --mode bnirl_subgoal \
      --dp-alpha 0.05
      
  done
done
