#!/bin/bash

#SBATCH --job-name mspacman_PPO_expert_KL_conv1_0.1_0 
#SBATCH --output=logs/slurmjob_%j.out
#SBATCH --error=logs/slurmjob_%j.err
#SBATCH --mail-user=asaran@cs.utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
###SBATCH --partition Test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/experts/mspacman_expert_conv1_KL_0 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/experts/mspacman_expert_conv1_KL_0.1 --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
