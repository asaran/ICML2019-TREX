#env_names = [['asterix','Asterix'],['centipede','Centipede'],['phoenix','Phoenix'],['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman']]
#env_names = [['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['enduro','Enduro'],['beamrider','BeamRider'],['qbert','Qbert'],['pong','Pong']]
#env_names = [['asterix','Asterix'],['phoenix','Phoenix'],['centipede','Centipede'],['seaquest','Seaquest'],['mspacman','MsPacman'],['breakout','Breakout']]
env_names = [['alien','Alien'],['asterix','Asterix'],['bank_heist','BankHeist'],['berzerk','Berzerk'],['breakout','Breakout'],['centipede','Centipede'],['demon_attack','DemonAttack'],['enduro','Enduro'],['freeway','Freeway'],['frostbite','Frostbite'],['hero','Hero'],['montezuma_revenge','MontezumaRevenge'],['mspacman','MsPacman'],['name_this_game','NameThisGame'],['phoneix','Phoenix'],['riverraid','Riverraid'],['road_runner','RoadRunner'],['seaquest','Seaquest'],['space_invaders','SpaceInvaders'],['venture','Venture']]

seeds = ['0']#,'1','2']
#conv_layer = '' #'1'
#user_type = 'expert'
gaze_loss_type = 'KL'
#gaze_reg = '0.01'
weights = ['0.001','0.005','0.01','0.05','0.1','0.3','0.5','0.7','0.9']

#gpu = ['0','1','2','3']
#gpu = ['0','1','2','3','4','5','6','7']
i=0

#import os
#if not os.path.isdir('tmp'):
#  os.mkdir('tmp')

#bash_file_name = 'tmp/PPO_'+user_type+'_conv'+conv_layer+'_'+gaze_loss_type+'.sh'
#f = open(bash_file_name,'w')
#f.write("#!/bin/bash\n")

for env in env_names:
  for seed in seeds:
    for weight in weights:
      #gpu_id = gpu[i%len(gpu)]
      screen_name = 'PPO_'+env[0]+'_'+gaze_loss_type+'_'+weight+'_seed'+seed+"_two-data"

      bash_file_name = 'gaze_pred/'+screen_name+'.sh'
      f = open(bash_file_name,'w')

      f.write("#!/bin/bash\n\n")
      f.write("#SBATCH --job-name "+screen_name+" \n")
      f.write("#SBATCH --output=logs/slurmjob_%j.out\n")
      f.write("#SBATCH --error=logs/slurmjob_%j.err\n")
      f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
      f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
      f.write("#SBATCH --partition titans\n")
      f.write("#SBATCH --nodes=1\n")
      f.write("#SBATCH --ntasks-per-node=1\n")
      f.write("#SBATCH --time 84:00:00\n") 
      f.write("#SBATCH --gres=gpu:1\n")
      f.write("#SBATCH --mem=50G\n")
      f.write("#SBATCH --cpus-per-task=8\n")
             
      f.write("OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=ppo_models/"+gaze_loss_type+"_two/"+env[0]+"_"+gaze_loss_type+"_"+weight+'_seed'+seed+" python -m baselines.run --alg=ppo2 --env="+env[1]+"NoFrameskip-v4 --custom_reward pytorch --custom_reward_path reward_models_AAAI2020/gazePred_"+env[0]+"_KL_"+weight+'_seed'+seed+"_two-data"+" --seed "+seed+" --num_timesteps=5e7 --save_interval=500 --num_env 9\n")
    
      i+=1
      
      f.close()
        
