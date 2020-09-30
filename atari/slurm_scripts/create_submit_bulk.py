import argparse

cmd_prefix = 'gaze_pred/PPO_'
weight = '0.05'
games = ['alien','asterix','bank_heist','berzerk','breakout','centipede','demon_attack','enduro','freeway','frostbite','hero','montezuma_revenge','mspacman','name_this_game','phoenix','riverraid','road_runner','seaquest','space_invaders','venture']
#games = ['alien','bank_heist','berzerk','demon_attack','enduro','freeway','frostbite','hero','montezuma_revenge','name_this_game','riverraid','road_runner','space_invaders','venture']
seed = '0'

#screen_name = game+'_'+cmd_prefix+seed+'_'+weight
f = open('bulk_submit.sh', 'w') 

f.write('#!/bin/bash\n\n\n')

for game in games:
	screen_name = game+'_PPO_'+seed+'_'+weight

	f.write('screen -dmS '+screen_name+' bash\n')
	f.write('screen -S '+screen_name+' -X stuff \"cd\n\"\n')
	f.write('screen -S '+screen_name+' -X stuff \"conda deactivate\n\"\n')
	f.write('screen -S '+screen_name+' -X stuff \"conda deactivate\n\"\n')
	f.write('screen -S '+screen_name+' -X stuff \". ./setup_trex.sh\n\"\n')
	f.write('screen -S '+screen_name+' -X stuff \"sbatch slurm_scripts/gaze_pred/PPO_'+game+'_KL_'+weight+'_seed'+seed+'_two-data.sh\n\"\n\n')
	#f.write('screen -dmS '+screen_name+' bash\n\n\n')

f.close()

