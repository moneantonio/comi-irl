#!/bin/bash

set -e  # Exit on error

echo "=== CoMI-IRL Finetuning Experiments ==="
echo "Start time: $(date)"

# for i in {0..2}
# do
#   echo "Running Pv4 experiment with seed $i"
#   python main.py -Pv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 3
#   # python main.py -Pv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 2
#   # python main.py -Pv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 1
# done

# for i in {0..2} 
# do
#   echo "Running Rv4 experiment with seed $i"
#   python main.py -Rv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 3
#   # python main.py -Rv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 2
#   # python main.py -Rv4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 1
# done

# for i in {0..2}
# do
#   echo "Running W2D experiment with seed $i"
#   # python main.py -W2D -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 2
#   python main.py -W2D -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 1
# done

for i in {0..2}
do
  echo "Running Ho4 experiment with seed $i"
  python main.py -Ho4 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 1
done

for i in {0..2}
do
  echo "Running HC5 experiment with seed $i"
  python main.py -HC5 -seed $i -airl --irl_training --parallel_irl --finetuning --num_unseen_modes 1
done

echo ""
echo "=== All experiments completed ==="
echo "End time: $(date)"