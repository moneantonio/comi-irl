#!/bin/bash

# no IRL training, we evaluate clustering performance

# for i in {0..9}
# do 
#   echo "Running Pv4 experiment with seed $i"
#   python main.py -Pv4 -seed $i -airl --beta 0.0 --gamma 0.0 --save_abl_results #C
#   python main.py -Pv4 -seed $i -airl --beta 0.0 --save_abl_results #CS
#   python main.py -Pv4 -seed $i -airl --gamma 0.0 --save_abl_results #DC
#   python main.py -Pv4 -seed $i -airl --save_abl_results #DCS
# done

for i in {0..9}
do
  echo "Running Rv4 experiment with seed $i"
  python main.py -Rv4 -seed $i -airl --beta 0.0 --gamma 0.0 --save_abl_results #C
  python main.py -Rv4 -seed $i -airl --beta 0.0 --save_abl_results #CS
  python main.py -Rv4 -seed $i -airl --gamma 0.0 --save_abl_results #DC
  python main.py -Rv4 -seed $i -airl --save_abl_results #DCS
done

for i in {0..9}
do
  echo "Running W2D experiment with seed $i"
  python main.py -W2D -seed $i -airl --beta 0.0 --gamma 0.0 --save_abl_results #C
  python main.py -W2D -seed $i -airl --beta 0.0 --save_abl_results #CS
  python main.py -W2D -seed $i -airl --gamma 0.0 --save_abl_results #DC
  python main.py -W2D -seed $i -airl --save_abl_results #DCS
done