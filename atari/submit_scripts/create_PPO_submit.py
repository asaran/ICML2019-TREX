
env_names = [['breakout','Breakout'], ['hero','Hero'],['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['enduro','Enduro'],['beamrider','BeamRider'],['qbert','Qbert'],['pong','Pong']]
seeds = ['0','1','2']
conv_layer = '1'
user_type = 'novice'
gaze_loss_type = 'KL'

gpu = ['0','1','3','5','6','7']
i=0


bash_file_name = 'PPO_novice_conv'+conv_layer+'_'+gaze_loss_type+'.sh'
f = open(bash_file_name,'w')
f.write("#!/bin/bash\n")

for env in env_names:
  for seed in seeds:
    gpu_id = gpu[i%len(gpu)]
    screen_name = env[0]+'_'+gaze_loss_type+'_'+seed
    #f.write("#!/bin/bash")
    f.write("screen -dmS "+screen_name+" bash\n")
    f.write("screen -S "+screen_name+" -X stuff \"cd\n")
    f.write("\"\n")
    f.write("screen -S "+screen_name+" -X stuff \". ./setup_trex.sh\n")
    f.write("\"\n")
                  
    f.write("screen -S "+screen_name+" -X stuff \"CUDA_VISIBLE_DEVICES="+gpu_id+" OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/"+user_type+"s/"+env[0]+"_novice_conv"+conv_layer+"_"+gaze_loss_type+"_"+seed+" python -m baselines.run --alg=ppo2 --env="+env[1]+"NoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+"_"+gaze_loss_type+" --seed "+seed+" --num_timesteps=5e7 --save_interval=500 --num_env 9\n")
    f.write("\"\n")
    i+=1
      
f.close()
        
