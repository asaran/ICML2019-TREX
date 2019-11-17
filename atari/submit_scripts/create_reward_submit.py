
#env_names = [['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders']]#,
#env_names = [['enduro','Enduro'],['beamrider','BeamRider'],['qbert','Qbert'],['pong','Pong']]
#env_names = [['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman']]
env_names = [['asterix','Asterix'], ['centipede','Centipede'], ['phoenix','Phoenix']]
seeds = ['0','1','2']
conv_layer = '2'
user_type = 'expert'
gaze_loss_type = 'quadratic'
data_dir = 'atari-head'

gpu = ['0','1','2','3','4','5','6','7']
i=0


bash_file_name = 'reward_'+user_type+'_conv'+conv_layer+'_'+gaze_loss_type+'.sh'
f = open(bash_file_name,'w')
f.write("#!/bin/bash\n")

for env in env_names:
  for seed in seeds:
    gpu_id = gpu[i%8]
    screen_name = env[0]+'_'+gaze_loss_type+'_'+seed+'_reward'
    #f.write("#!/bin/bash")
    f.write("screen -dmS "+screen_name+" bash\n")
    f.write("screen -S "+screen_name+" -X stuff \"cd\n")
    f.write("\"\n")
    f.write("screen -S "+screen_name+" -X stuff \". ./setup_trex.sh\n")
    f.write("\"\n")
                  
    f.write("screen -S "+screen_name+" -X stuff \"CUDA_VISIBLE_DEVICES="+gpu_id+" python LearnAtariRewardGaze.py --data_dir ../../learning-rewards-of-learners/data/"+data_dir+"/ --gaze_loss "+gaze_loss_type+" --gaze_conv_layer "+conv_layer+" --env_name "+env[0]+" --reward_model_path learned_models/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+"_new\n")
    f.write("\"\n")
    i+=1
      
f.close()
        
