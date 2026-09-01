#!/bin/bash

# ***********************************G-GAIL experiments
# for i in {0..4}
# do
#   echo "*************** Running G-GAIL baselines Reacher experiment with seed $i"
#   python khgail.py --env Reacher-v4 --seed $i --clusterer leiden --irl gail --parallel
#   python khgail.py --env Reacher-v4 --seed $i --clusterer leiden --irl airl --parallel
# done

# for i in {0..4}
# do
#   echo "*************** Running G-GAIL baselines Pusher experiment with seed $i"
#   python khgail.py --env Pusher-v4 --seed $i --clusterer leiden --irl gail --parallel
#   python khgail.py --env Pusher-v4 --seed $i --clusterer leiden --irl airl --parallel
# done

# for i in {0..4}
# do
#   echo "*************** Running G-GAIL baselines Walker2d experiment with seed $i"
#   python khgail.py --env Walker2d-v4 --seed $i --clusterer leiden --irl gail --parallel
#   python khgail.py --env Walker2d-v4 --seed $i --clusterer leiden --irl airl --parallel
# done


# for i in {0..4}
# do
#   echo "*************** Running G-GAIL baselines Hopper-v4 experiment with seed $i"

#   python khgail.py --env Hopper-v4 --seed $i --clusterer leiden --irl airl --parallel

#   python khgail.py --env Hopper-v4 --seed $i --clusterer leiden --irl gail --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running G-GAIL baselines HalfCheetah-v5 experiment with seed $i"

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer leiden --irl airl --parallel

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer leiden --irl gail --parallel

# done

# ********************************************K-GAIL experiments Reacher

# for i in {0..4}
# do
#   echo "*************** Running baselines Reacher experiment with seed $i and k=3"

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 3 --irl gail --parallel

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 3 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines Reacher experiment with seed $i and k=6"

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 6 --irl gail --parallel

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 6 --irl airl --parallel

# done


# for i in {0..4}
# do
#   echo "*************** Running baselines Reacher experiment with seed $i and k=12"

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 12 --irl gail --parallel

#   python khgail.py --env Reacher-v4 --seed $i --clusterer kmeans --k 12 --irl airl --parallel

# done


# K-GAIL experiments Pusher

# for i in {0..4}
# do
#   echo "*************** Running baselines Pusher experiment with seed $i and k=3"

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 3 --irl gail --parallel

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 3 --irl airl --parallel

# done


# for i in {0..4}
# do
#   echo "*************** Running baselines Pusher experiment with seed $i and k=6"

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 6 --irl gail --parallel

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 6 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines Pusher experiment with seed $i and k=12"

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 12 --irl gail --parallel

#   python khgail.py --env Pusher-v4 --seed $i --clusterer kmeans --k 12 --irl airl --parallel

# done


# *****************************************K-GAIL experiments Walker


# for i in {0..4}
# do
#   echo "*************** Running baselines Walker2d experiment with seed $i and k=3"

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 3 --irl gail --parallel

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 3 --irl airl --parallel

# done


# for i in {0..4}
# do
#   echo "*************** Running baselines Walker2d experiment with seed $i and k=6"

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 6 --irl gail --parallel

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 6 --irl airl --parallel

# done



# for i in {0..4}
# do
#   echo "*************** Running baselines Walker2d experiment with seed $i and k=12"

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 12 --irl gail --parallel

#   python khgail.py --env Walker2d-v4 --seed $i --clusterer kmeans --k 12 --irl airl --parallel

# done



# *****************************************K- experiments Hopper


# for i in {0..4}
# do
#   echo "*************** Running baselines Hopper experiment with seed $i and k=3"

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 3 --irl gail --parallel

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 3 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines Hopper experiment with seed $i and k=6"

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 6 --irl gail --parallel

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 6 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines Hopper experiment with seed $i and k=12"

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 12 --irl gail --parallel

#   python khgail.py --env Hopper-v4 --seed $i --clusterer kmeans --k 12 --irl airl --parallel

# done


# *****************************************K- experiments HalfCheetah


# for i in {0..4}
# do
#   echo "*************** Running baselines HalfCheetah experiment with seed $i and k=3"

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 3 --irl gail --parallel

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 3 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines HalfCheetah experiment with seed $i and k=6"

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 6 --irl gail --parallel

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 6 --irl airl --parallel

# done

# for i in {0..4}
# do
#   echo "*************** Running baselines HalfCheetah experiment with seed $i and k=12"

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 12 --irl gail --parallel

#   python khgail.py --env HalfCheetah-v5 --seed $i --clusterer kmeans --k 12 --irl airl --parallel

# done
