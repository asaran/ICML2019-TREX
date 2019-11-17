#!/bin/bash
screen -dmS enduro_sinkhorn_0_reward bash
screen -S enduro_sinkhorn_0_reward -X stuff "cd
"
screen -S enduro_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name enduro --reward_model_path learned_models/novices/enduro_novice_conv2_sinkhorn
"
screen -dmS enduro_sinkhorn_1_reward bash
screen -S enduro_sinkhorn_1_reward -X stuff "cd
"
screen -S enduro_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name enduro --reward_model_path learned_models/novices/enduro_novice_conv2_sinkhorn
"
screen -dmS enduro_sinkhorn_2_reward bash
screen -S enduro_sinkhorn_2_reward -X stuff "cd
"
screen -S enduro_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name enduro --reward_model_path learned_models/novices/enduro_novice_conv2_sinkhorn
"
screen -dmS beamrider_sinkhorn_0_reward bash
screen -S beamrider_sinkhorn_0_reward -X stuff "cd
"
screen -S beamrider_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name beamrider --reward_model_path learned_models/novices/beamrider_novice_conv2_sinkhorn
"
screen -dmS beamrider_sinkhorn_1_reward bash
screen -S beamrider_sinkhorn_1_reward -X stuff "cd
"
screen -S beamrider_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name beamrider --reward_model_path learned_models/novices/beamrider_novice_conv2_sinkhorn
"
screen -dmS beamrider_sinkhorn_2_reward bash
screen -S beamrider_sinkhorn_2_reward -X stuff "cd
"
screen -S beamrider_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name beamrider --reward_model_path learned_models/novices/beamrider_novice_conv2_sinkhorn
"
screen -dmS qbert_sinkhorn_0_reward bash
screen -S qbert_sinkhorn_0_reward -X stuff "cd
"
screen -S qbert_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name qbert --reward_model_path learned_models/novices/qbert_novice_conv2_sinkhorn
"
screen -dmS qbert_sinkhorn_1_reward bash
screen -S qbert_sinkhorn_1_reward -X stuff "cd
"
screen -S qbert_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name qbert --reward_model_path learned_models/novices/qbert_novice_conv2_sinkhorn
"
screen -dmS qbert_sinkhorn_2_reward bash
screen -S qbert_sinkhorn_2_reward -X stuff "cd
"
screen -S qbert_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name qbert --reward_model_path learned_models/novices/qbert_novice_conv2_sinkhorn
"
screen -dmS pong_sinkhorn_0_reward bash
screen -S pong_sinkhorn_0_reward -X stuff "cd
"
screen -S pong_sinkhorn_0_reward -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_0_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name pong --reward_model_path learned_models/novices/pong_novice_conv2_sinkhorn
"
screen -dmS pong_sinkhorn_1_reward bash
screen -S pong_sinkhorn_1_reward -X stuff "cd
"
screen -S pong_sinkhorn_1_reward -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_1_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name pong --reward_model_path learned_models/novices/pong_novice_conv2_sinkhorn
"
screen -dmS pong_sinkhorn_2_reward bash
screen -S pong_sinkhorn_2_reward -X stuff "cd
"
screen -S pong_sinkhorn_2_reward -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_2_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/novice-atari-head/ --gaze_loss sinkhorn --gaze_conv_layer 2 --env_name pong --reward_model_path learned_models/novices/pong_novice_conv2_sinkhorn
"
