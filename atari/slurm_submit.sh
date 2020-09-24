#!/bin/bash

env_names='alien asterix bank_heist berzerk breakout centipede demon_attack enduro freeway frostbite hero montezuma_revenge mspacman name_this_game phoenix riverraid road_runner seaquest space_invaders venture'

cmd = 'sbatch '
prefix = 'slurm_scripts/gaze_pred/eval_'
suffix = '_base_seed0_30000.sh'

for env in $env_names
do
  echo $env
  "$cmd$prefix$env$suffix"
done

echo All done
