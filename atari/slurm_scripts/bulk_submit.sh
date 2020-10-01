#!/bin/bash


screen -dmS alien_PPO_0_0.05 bash
screen -S alien_PPO_0_0.05 -X stuff "cd
"
screen -S alien_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S alien_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S alien_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S alien_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_alien_KL_0.05_seed0_two-data.sh
"

screen -dmS asterix_PPO_0_0.05 bash
screen -S asterix_PPO_0_0.05 -X stuff "cd
"
screen -S asterix_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S asterix_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S asterix_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S asterix_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_asterix_KL_0.05_seed0_two-data.sh
"

screen -dmS bank_heist_PPO_0_0.05 bash
screen -S bank_heist_PPO_0_0.05 -X stuff "cd
"
screen -S bank_heist_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S bank_heist_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S bank_heist_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S bank_heist_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_bank_heist_KL_0.05_seed0_two-data.sh
"

screen -dmS berzerk_PPO_0_0.05 bash
screen -S berzerk_PPO_0_0.05 -X stuff "cd
"
screen -S berzerk_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S berzerk_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S berzerk_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S berzerk_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_berzerk_KL_0.05_seed0_two-data.sh
"

screen -dmS breakout_PPO_0_0.05 bash
screen -S breakout_PPO_0_0.05 -X stuff "cd
"
screen -S breakout_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S breakout_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S breakout_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S breakout_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_breakout_KL_0.05_seed0_two-data.sh
"

screen -dmS centipede_PPO_0_0.05 bash
screen -S centipede_PPO_0_0.05 -X stuff "cd
"
screen -S centipede_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S centipede_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S centipede_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S centipede_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_centipede_KL_0.05_seed0_two-data.sh
"

screen -dmS demon_attack_PPO_0_0.05 bash
screen -S demon_attack_PPO_0_0.05 -X stuff "cd
"
screen -S demon_attack_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S demon_attack_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S demon_attack_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S demon_attack_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_demon_attack_KL_0.05_seed0_two-data.sh
"

screen -dmS enduro_PPO_0_0.05 bash
screen -S enduro_PPO_0_0.05 -X stuff "cd
"
screen -S enduro_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S enduro_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S enduro_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S enduro_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_enduro_KL_0.05_seed0_two-data.sh
"

screen -dmS freeway_PPO_0_0.05 bash
screen -S freeway_PPO_0_0.05 -X stuff "cd
"
screen -S freeway_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S freeway_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S freeway_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S freeway_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_freeway_KL_0.05_seed0_two-data.sh
"

screen -dmS frostbite_PPO_0_0.05 bash
screen -S frostbite_PPO_0_0.05 -X stuff "cd
"
screen -S frostbite_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S frostbite_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S frostbite_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S frostbite_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_frostbite_KL_0.05_seed0_two-data.sh
"

screen -dmS hero_PPO_0_0.05 bash
screen -S hero_PPO_0_0.05 -X stuff "cd
"
screen -S hero_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S hero_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S hero_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S hero_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_hero_KL_0.05_seed0_two-data.sh
"

screen -dmS montezuma_revenge_PPO_0_0.05 bash
screen -S montezuma_revenge_PPO_0_0.05 -X stuff "cd
"
screen -S montezuma_revenge_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S montezuma_revenge_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S montezuma_revenge_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S montezuma_revenge_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_montezuma_revenge_KL_0.05_seed0_two-data.sh
"

screen -dmS mspacman_PPO_0_0.05 bash
screen -S mspacman_PPO_0_0.05 -X stuff "cd
"
screen -S mspacman_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S mspacman_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S mspacman_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S mspacman_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_mspacman_KL_0.05_seed0_two-data.sh
"

screen -dmS name_this_game_PPO_0_0.05 bash
screen -S name_this_game_PPO_0_0.05 -X stuff "cd
"
screen -S name_this_game_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S name_this_game_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S name_this_game_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S name_this_game_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_name_this_game_KL_0.05_seed0_two-data.sh
"

screen -dmS phoenix_PPO_0_0.05 bash
screen -S phoenix_PPO_0_0.05 -X stuff "cd
"
screen -S phoenix_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S phoenix_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S phoenix_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S phoenix_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_phoenix_KL_0.05_seed0_two-data.sh
"

screen -dmS riverraid_PPO_0_0.05 bash
screen -S riverraid_PPO_0_0.05 -X stuff "cd
"
screen -S riverraid_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S riverraid_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S riverraid_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S riverraid_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_riverraid_KL_0.05_seed0_two-data.sh
"

screen -dmS road_runner_PPO_0_0.05 bash
screen -S road_runner_PPO_0_0.05 -X stuff "cd
"
screen -S road_runner_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S road_runner_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S road_runner_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S road_runner_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_road_runner_KL_0.05_seed0_two-data.sh
"

screen -dmS seaquest_PPO_0_0.05 bash
screen -S seaquest_PPO_0_0.05 -X stuff "cd
"
screen -S seaquest_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S seaquest_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S seaquest_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S seaquest_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_seaquest_KL_0.05_seed0_two-data.sh
"

screen -dmS space_invaders_PPO_0_0.05 bash
screen -S space_invaders_PPO_0_0.05 -X stuff "cd
"
screen -S space_invaders_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S space_invaders_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S space_invaders_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S space_invaders_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_space_invaders_KL_0.05_seed0_two-data.sh
"

screen -dmS venture_PPO_0_0.05 bash
screen -S venture_PPO_0_0.05 -X stuff "cd
"
screen -S venture_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S venture_PPO_0_0.05 -X stuff "conda deactivate
"
screen -S venture_PPO_0_0.05 -X stuff ". ./setup_trex.sh
"
screen -S venture_PPO_0_0.05 -X stuff "sbatch slurm_scripts/gaze_pred/PPO_venture_KL_0.05_seed0_two-data.sh
"

