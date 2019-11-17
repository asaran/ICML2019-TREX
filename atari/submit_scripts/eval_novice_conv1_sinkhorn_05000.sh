#!/bin/bash
screen -dmS breakout_sinkhorn_0_05000 bash
screen -S breakout_sinkhorn_0_05000 -X stuff "cd
"
screen -S breakout_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S breakout_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name breakout --checkpointpath path_to_logs/novices/breakout_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS hero_sinkhorn_0_05000 bash
screen -S hero_sinkhorn_0_05000 -X stuff "cd
"
screen -S hero_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S hero_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name hero --checkpointpath path_to_logs/novices/hero_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS seaquest_sinkhorn_0_05000 bash
screen -S seaquest_sinkhorn_0_05000 -X stuff "cd
"
screen -S seaquest_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S seaquest_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name seaquest --checkpointpath path_to_logs/novices/seaquest_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS spaceinvaders_sinkhorn_0_05000 bash
screen -S spaceinvaders_sinkhorn_0_05000 -X stuff "cd
"
screen -S spaceinvaders_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S spaceinvaders_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name spaceinvaders --checkpointpath path_to_logs/novices/spaceinvaders_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS enduro_sinkhorn_0_05000 bash
screen -S enduro_sinkhorn_0_05000 -X stuff "cd
"
screen -S enduro_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S enduro_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name enduro --checkpointpath path_to_logs/novices/enduro_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS beamrider_sinkhorn_0_05000 bash
screen -S beamrider_sinkhorn_0_05000 -X stuff "cd
"
screen -S beamrider_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S beamrider_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name beamrider --checkpointpath path_to_logs/novices/beamrider_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS qbert_sinkhorn_0_05000 bash
screen -S qbert_sinkhorn_0_05000 -X stuff "cd
"
screen -S qbert_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S qbert_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name qbert --checkpointpath path_to_logs/novices/qbert_novice_conv1_sinkhorn_0/checkpoints/05000
"
screen -dmS pong_sinkhorn_0_05000 bash
screen -S pong_sinkhorn_0_05000 -X stuff "cd
"
screen -S pong_sinkhorn_0_05000 -X stuff ". ./setup_trex.sh
"
screen -S pong_sinkhorn_0_05000 -X stuff "python evaluateLearnedPolicy.py --env_name pong --checkpointpath path_to_logs/novices/pong_novice_conv1_sinkhorn_0/checkpoints/05000
"
