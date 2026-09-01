#!/bin/bash

for i in {0..4}
do
  echo "Running Pv4 experiment with seed $i"
  python main.py -Pv4 -seed $i -airl --irl_training --parallel_irl
  python main.py -Pv4 -seed $i -gail --irl_training --parallel_irl
done

for i in {0..4} 
do
  echo "Running Rv4 experiment with seed $i"
  python main.py -Rv4 -seed $i -airl --irl_training --parallel_irl
  python main.py -Rv4 -seed $i -gail --irl_training --parallel_irl
done

for i in {0..4}
do
  echo "Running W2D experiment with seed $i"

  python main.py -W2D -seed $i -airl --irl_training --parallel_irl

  python main.py -W2D -seed $i -gail --irl_training --parallel_irl

done

for i in {0..4}
do

  echo "Running Ho4 experiment with seed $i"

  python main.py -Ho4 -seed $i -airl --irl_training --parallel_irl

  python main.py -Ho4 -seed $i -gail --irl_training --parallel_irl

done

for i in {0..4}
do

  echo "Running HC5 experiment with seed $i"

  python main.py -HC5 -seed $i -airl --irl_training --parallel_irl

  python main.py -HC5 -seed $i -gail --irl_training --parallel_irl

done