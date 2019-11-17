#!/bin/bash
screen -dmS breakout_exact_0_reward bash
screen -S breakout_exact_0_reward -X stuff "cd
"
screen -S breakout_exact_0_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_exact_0_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/experts/breakout_expert_conv2_exact
"
screen -dmS breakout_exact_1_reward bash
screen -S breakout_exact_1_reward -X stuff "cd
"
screen -S breakout_exact_1_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_exact_1_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/experts/breakout_expert_conv2_exact
"
screen -dmS breakout_exact_2_reward bash
screen -S breakout_exact_2_reward -X stuff "cd
"
screen -S breakout_exact_2_reward -X stuff ". ./setup_trex.sh
"
screen -S breakout_exact_2_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name breakout --reward_model_path learned_models/experts/breakout_expert_conv2_exact
"
screen -dmS hero_exact_0_reward bash
screen -S hero_exact_0_reward -X stuff "cd
"
screen -S hero_exact_0_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_exact_0_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/experts/hero_expert_conv2_exact
"
screen -dmS hero_exact_1_reward bash
screen -S hero_exact_1_reward -X stuff "cd
"
screen -S hero_exact_1_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_exact_1_reward -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/experts/hero_expert_conv2_exact
"
screen -dmS hero_exact_2_reward bash
screen -S hero_exact_2_reward -X stuff "cd
"
screen -S hero_exact_2_reward -X stuff ". ./setup_trex.sh
"
screen -S hero_exact_2_reward -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name hero --reward_model_path learned_models/experts/hero_expert_conv2_exact
"
screen -dmS seaquest_exact_0_reward bash
screen -S seaquest_exact_0_reward -X stuff "cd
"
screen -S seaquest_exact_0_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_exact_0_reward -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/experts/seaquest_expert_conv2_exact
"
screen -dmS seaquest_exact_1_reward bash
screen -S seaquest_exact_1_reward -X stuff "cd
"
screen -S seaquest_exact_1_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_exact_1_reward -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/experts/seaquest_expert_conv2_exact
"
screen -dmS seaquest_exact_2_reward bash
screen -S seaquest_exact_2_reward -X stuff "cd
"
screen -S seaquest_exact_2_reward -X stuff ". ./setup_trex.sh
"
screen -S seaquest_exact_2_reward -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name seaquest --reward_model_path learned_models/experts/seaquest_expert_conv2_exact
"
screen -dmS spaceinvaders_exact_0_reward bash
screen -S spaceinvaders_exact_0_reward -X stuff "cd
"
screen -S spaceinvaders_exact_0_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_exact_0_reward -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/experts/spaceinvaders_expert_conv2_exact
"
screen -dmS spaceinvaders_exact_1_reward bash
screen -S spaceinvaders_exact_1_reward -X stuff "cd
"
screen -S spaceinvaders_exact_1_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_exact_1_reward -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/experts/spaceinvaders_expert_conv2_exact
"
screen -dmS spaceinvaders_exact_2_reward bash
screen -S spaceinvaders_exact_2_reward -X stuff "cd
"
screen -S spaceinvaders_exact_2_reward -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_exact_2_reward -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name spaceinvaders --reward_model_path learned_models/experts/spaceinvaders_expert_conv2_exact
"
screen -dmS mspacman_exact_0_reward bash
screen -S mspacman_exact_0_reward -X stuff "cd
"
screen -S mspacman_exact_0_reward -X stuff ". ./setup_trex.sh
"
screen -S mspacman_exact_0_reward -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name mspacman --reward_model_path learned_models/experts/mspacman_expert_conv2_exact
"
screen -dmS mspacman_exact_1_reward bash
screen -S mspacman_exact_1_reward -X stuff "cd
"
screen -S mspacman_exact_1_reward -X stuff ". ./setup_trex.sh
"
screen -S mspacman_exact_1_reward -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name mspacman --reward_model_path learned_models/experts/mspacman_expert_conv2_exact
"
screen -dmS mspacman_exact_2_reward bash
screen -S mspacman_exact_2_reward -X stuff "cd
"
screen -S mspacman_exact_2_reward -X stuff ". ./setup_trex.sh
"
screen -S mspacman_exact_2_reward -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/atari-head/ --gaze_loss exact --gaze_conv_layer 2 --env_name mspacman --reward_model_path learned_models/experts/mspacman_expert_conv2_exact
"
