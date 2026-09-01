#!/bin/bash

python essinfogail/train_classifier_K.py --env_id Reacher-v4 --data_num_modes 6 --epochs 100 --short --latent_k 3
python essinfogail/train_classifier_K.py --env_id Reacher-v4 --data_num_modes 6 --epochs 100 --short --latent_k 6
python essinfogail/train_classifier_K.py --env_id Reacher-v4 --data_num_modes 12 --epochs 100 --short --latent_k 12

python essinfogail/train_classifier_K.py --env_id Pusher-v4 --data_num_modes 6 --epochs 100 --short --latent_k 3
python essinfogail/train_classifier_K.py --env_id Pusher-v4 --data_num_modes 6 --epochs 100 --short --latent_k 6
python essinfogail/train_classifier_K.py --env_id Pusher-v4 --data_num_modes 12 --epochs 100 --short --latent_k 12

python essinfogail/train_classifier_K.py --env_id Walker2d-v4 --data_num_modes 3 --epochs 100 --short --latent_k 3
python essinfogail/train_classifier_K.py --env_id Walker2d-v4 --data_num_modes 6 --epochs 100 --short --latent_k 6
python essinfogail/train_classifier_K.py --env_id Walker2d-v4 --data_num_modes 12 --epochs 100 --short --latent_k 12

python essinfogail/train_classifier_K.py --env_id Hopper-v4 --data_num_modes 3 --epochs 20 --short --latent_k 3
python essinfogail/train_classifier_K.py --env_id Hopper-v4 --data_num_modes 6 --epochs 20 --short --latent_k 6
python essinfogail/train_classifier_K.py --env_id Hopper-v4 --data_num_modes 12 --epochs 20 --short --latent_k 12

python essinfogail/train_classifier_K.py --env_id HalfCheetah-v5 --data_num_modes 3 --epochs 20 --short --latent_k 3
python essinfogail/train_classifier_K.py --env_id HalfCheetah-v5 --data_num_modes 6 --epochs 20 --short --latent_k 6
python essinfogail/train_classifier_K.py --env_id HalfCheetah-v5 --data_num_modes 12 --epochs 20 --short --latent_k 12