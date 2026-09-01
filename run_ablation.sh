#!/bin/bash

#first run no IRL training

for i in {0..2}
do
  echo "Running Pv4 experiment with seed $i"
  python main.py -Pv4 -seed $i -airl --ablation
  # python main.py -Pv4 -seed $i -airl --irl_training --parallel_irl --ablation
done

for i in {0..2} 
do
  echo "Running Rv4 experiment with seed $i"
  python main.py -Rv4 -seed $i -airl --ablation
  # python main.py -Rv4 -seed $i -airl --irl_training --parallel_irl --ablation
done

for i in {0..2}
do
  echo "Running W2D experiment with seed $i"
  python main.py -W2D -seed $i -airl --ablation
  # python main.py -W2D -seed $i -airl --irl_training --parallel_irl --ablation
done