#!/bin/bash
screen -dmS breakout_sinkhorn_0_reward bash
screen -S breakout_sinkhorn_0_reward -X stuff "cd
"
screen -S breakout_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/novices/breakout_novice_conv2_sinkhorn
"
screen -dmS breakout_sinkhorn_1_reward bash
screen -S breakout_sinkhorn_1_reward -X stuff "cd
"
screen -S breakout_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/novices/breakout_novice_conv2_sinkhorn
"
screen -dmS breakout_sinkhorn_2_reward bash
screen -S breakout_sinkhorn_2_reward -X stuff "cd
"
screen -S breakout_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/novices/breakout_novice_conv2_sinkhorn
"
screen -dmS hero_sinkhorn_0_reward bash
screen -S hero_sinkhorn_0_reward -X stuff "cd
"
screen -S hero_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/novices/hero_novice_conv2_sinkhorn
"
screen -dmS hero_sinkhorn_1_reward bash
screen -S hero_sinkhorn_1_reward -X stuff "cd
"
screen -S hero_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/novices/hero_novice_conv2_sinkhorn
"
screen -dmS hero_sinkhorn_2_reward bash
screen -S hero_sinkhorn_2_reward -X stuff "cd
"
screen -S hero_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/novices/hero_novice_conv2_sinkhorn
"
screen -dmS seaquest_sinkhorn_0_reward bash
screen -S seaquest_sinkhorn_0_reward -X stuff "cd
"
screen -S seaquest_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/novices/seaquest_novice_conv2_sinkhorn
"
screen -dmS seaquest_sinkhorn_1_reward bash
screen -S seaquest_sinkhorn_1_reward -X stuff "cd
"
screen -S seaquest_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/novices/seaquest_novice_conv2_sinkhorn
"
screen -dmS seaquest_sinkhorn_2_reward bash
screen -S seaquest_sinkhorn_2_reward -X stuff "cd
"
screen -S seaquest_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/novices/seaquest_novice_conv2_sinkhorn
"
screen -dmS spaceinvaders_sinkhorn_0_reward bash
screen -S spaceinvaders_sinkhorn_0_reward -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn
"
screen -dmS spaceinvaders_sinkhorn_1_reward bash
screen -S spaceinvaders_sinkhorn_1_reward -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn
"
screen -dmS spaceinvaders_sinkhorn_2_reward bash
screen -S spaceinvaders_sinkhorn_2_reward -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn
"
