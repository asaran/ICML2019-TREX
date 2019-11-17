#!/bin/bash
screen -dmS breakout_sinkhorn_0 bash
screen -S breakout_sinkhorn_0 -X stuff "cd
"
screen -S breakout_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/breakout_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/breakout_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS breakout_sinkhorn_1 bash
screen -S breakout_sinkhorn_1 -X stuff "cd
"
screen -S breakout_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/breakout_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/breakout_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS breakout_sinkhorn_2 bash
screen -S breakout_sinkhorn_2 -X stuff "cd
"
screen -S breakout_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/breakout_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/breakout_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS hero_sinkhorn_0 bash
screen -S hero_sinkhorn_0 -X stuff "cd
"
screen -S hero_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/hero_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/hero_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS hero_sinkhorn_1 bash
screen -S hero_sinkhorn_1 -X stuff "cd
"
screen -S hero_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/hero_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/hero_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS hero_sinkhorn_2 bash
screen -S hero_sinkhorn_2 -X stuff "cd
"
screen -S hero_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/hero_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/hero_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS seaquest_sinkhorn_0 bash
screen -S seaquest_sinkhorn_0 -X stuff "cd
"
screen -S seaquest_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/seaquest_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/seaquest_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS seaquest_sinkhorn_1 bash
screen -S seaquest_sinkhorn_1 -X stuff "cd
"
screen -S seaquest_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/seaquest_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/seaquest_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS seaquest_sinkhorn_2 bash
screen -S seaquest_sinkhorn_2 -X stuff "cd
"
screen -S seaquest_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/seaquest_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/seaquest_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS spaceinvaders_sinkhorn_0 bash
screen -S spaceinvaders_sinkhorn_0 -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/spaceinvaders_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS spaceinvaders_sinkhorn_1 bash
screen -S spaceinvaders_sinkhorn_1 -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/spaceinvaders_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS spaceinvaders_sinkhorn_2 bash
screen -S spaceinvaders_sinkhorn_2 -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/spaceinvaders_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/spaceinvaders_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS enduro_sinkhorn_0 bash
screen -S enduro_sinkhorn_0 -X stuff "cd
"
screen -S enduro_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/enduro_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=EnduroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/enduro_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS enduro_sinkhorn_1 bash
screen -S enduro_sinkhorn_1 -X stuff "cd
"
screen -S enduro_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/enduro_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=EnduroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/enduro_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS enduro_sinkhorn_2 bash
screen -S enduro_sinkhorn_2 -X stuff "cd
"
screen -S enduro_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/enduro_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=EnduroNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/enduro_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS beamrider_sinkhorn_0 bash
screen -S beamrider_sinkhorn_0 -X stuff "cd
"
screen -S beamrider_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/beamrider_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=BeamRiderNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/beamrider_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS beamrider_sinkhorn_1 bash
screen -S beamrider_sinkhorn_1 -X stuff "cd
"
screen -S beamrider_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/beamrider_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=BeamRiderNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/beamrider_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS beamrider_sinkhorn_2 bash
screen -S beamrider_sinkhorn_2 -X stuff "cd
"
screen -S beamrider_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/beamrider_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=BeamRiderNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/beamrider_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS qbert_sinkhorn_0 bash
screen -S qbert_sinkhorn_0 -X stuff "cd
"
screen -S qbert_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/qbert_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=QbertNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/qbert_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS qbert_sinkhorn_1 bash
screen -S qbert_sinkhorn_1 -X stuff "cd
"
screen -S qbert_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/qbert_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=QbertNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/qbert_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS qbert_sinkhorn_2 bash
screen -S qbert_sinkhorn_2 -X stuff "cd
"
screen -S qbert_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/qbert_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=QbertNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/qbert_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS pong_sinkhorn_0 bash
screen -S pong_sinkhorn_0 -X stuff "cd
"
screen -S pong_sinkhorn_0 -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_0 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/pong_novice_conv2_sinkhorn_0 python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/pong_novice_conv2_sinkhorn --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS pong_sinkhorn_1 bash
screen -S pong_sinkhorn_1 -X stuff "cd
"
screen -S pong_sinkhorn_1 -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_1 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/pong_novice_conv2_sinkhorn_1 python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/pong_novice_conv2_sinkhorn --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
screen -dmS pong_sinkhorn_2 bash
screen -S pong_sinkhorn_2 -X stuff "cd
"
screen -S pong_sinkhorn_2 -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_2 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/novices/pong_novice_conv2_sinkhorn_2 python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/novices/pong_novice_conv2_sinkhorn --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
"
