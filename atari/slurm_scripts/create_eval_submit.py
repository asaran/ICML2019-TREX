env_names = [['asterix','Asterix'],['centipede','Centipede'],['phoenix','Phoenix'],['breakout','Breakout'],['seaquest','Seaquest'],['mspacman','MsPacman']]
#env_names = [['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['enduro','Enduro'],['beamrider','BeamRider'],['qbert','Qbert'],['pong','Pong']]
#env_names = [['asterix','Asterix'], ['berzerk','Berzerk'],['centipede','Centipede']]
seeds = ['0']
conv_layer = '' #'1'
#user_type = 'expert'
gaze_loss_type = ''#'KL'
gaze_reg = ''#'0.01'

models = {'asterix':'path_to_logs/experts/asterix_expert_1',
	'breakout':'path_to_logs/experts/breakout_expert_0',
	'centipede':'path_to_logs/experts/centipede_expert_0',
	'phoenix':'path_to_logs/experts/phoenix_expert_1',
	'mspacman':'path_to_logs/experts/mspacman_expert_0',
	'seaquest':'path_to_logs/experts/seaquest_expert_0'}
models_KL = {
	'asterix':'path_to_logs/experts_KL/asterix_expert_conv1_KL_1',
	'breakout':'path_to_logs/experts_KL/breakout_expert_conv1_KL_0.1_0',
	'centipede':'path_to_logs/experts_KL/centipede_expert_conv1_KL_0.1_0',
	'phoenix':'path_to_logs/experts_KL/phoenix_expert_conv1_KL_0.1_0',
	'mspacman':'path_to_logs/experts_KL/mspacman_expert_conv1_KL_0.1_0',
	'seaquest':'path_to_logs/experts_KL/seaquest_expert_conv1_KL_0.01_0'
}

#gpu = ['0','1','2','3']
#gpu = ['0','1','2','3','4','5','6','7']
i=0

import os
if not os.path.isdir('tmp'):
  os.mkdir('tmp')

#bash_file_name = 'tmp/PPO_'+user_type+'_conv'+conv_layer+'_'+gaze_loss_type+'.sh'
#f = open(bash_file_name,'w')
#f.write("#!/bin/bash\n")

for env in env_names:
    #gpu_id = gpu[i%len(gpu)]
    screen_name = env[0]+'_eval'

    bash_file_name = 'tmp/'+screen_name+'.sh'
    f = open(bash_file_name,'w')

    f.write("#!/bin/bash\n\n")
    f.write("#SBATCH --job-name "+screen_name+" \n")
    f.write("#SBATCH --output=logs/slurmjob_%j.out\n")
    f.write("#SBATCH --error=logs/slurmjob_%j.err\n")
    f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
    f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
    f.write("###SBATCH --partition Test\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks-per-node=1\n")
    f.write("#SBATCH --time 72:00:00\n") 
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --mem=50G\n")
    f.write("#SBATCH --cpus-per-task=8\n")
             
    #f.write("OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+"_"+gaze_reg+'_'+seed+" python -m baselines.run --alg=ppo2 --env="+env[1]+"NoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+"_"+gaze_reg+" --seed "+seed+" --num_timesteps=5e7 --save_interval=500 --num_env 9\n")
    f.write("python evaluateLearnedPolicy.py --env_name "+env[0]+" --checkpointpath "+models[env[0]]+'/checkpoints/43000')
    i+=1
      
    f.close()
        


for env in env_names:
    #gpu_id = gpu[i%len(gpu)]
    screen_name = env[0]+'_eval_KL'

    bash_file_name = 'tmp/'+screen_name+'.sh'
    f = open(bash_file_name,'w')

    f.write("#!/bin/bash\n\n")
    f.write("#SBATCH --job-name "+screen_name+" \n")
    f.write("#SBATCH --output=logs/slurmjob_%j.out\n")
    f.write("#SBATCH --error=logs/slurmjob_%j.err\n")
    f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
    f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
    f.write("###SBATCH --partition Test\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks-per-node=1\n")
    f.write("#SBATCH --time 72:00:00\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --mem=50G\n")
    f.write("#SBATCH --cpus-per-task=8\n")

    #f.write("OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+"_"+gaze_reg+'_'+seed+" python -m baselines.run --alg=ppo2 --env="+env[1]+"NoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+"_"+gaze_reg+" --seed "+seed+" --num_timesteps=5e7 --save_interval=500 --num_env 9\n")
    f.write("python evaluateLearnedPolicy.py --env_name "+env[0]+" --checkpointpath "+models_KL[env[0]]+'/checkpoints/43000')
    i+=1

    f.close()

