env_names = [['hero','Hero'],['seaquest','Seaquest'],['breakout','Breakout'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['phoenix','Phoenix'],['asterix','Asterix'],['centipede','Centipede']]
#env_names = [['hero','Hero'],['seaquest','Seaquest'],['enduro','Enduro'],['beamrider','BeamRider'],['pong','Pong'],['breakout','Breakout'],['spaceinvaders','SpaceInvaders'],['qbert','Qbert']]
#env_names =[['asterix','Asterix'],['berzerk','Berzerk']]
seeds = ['0']#,'1','2']
conv_layer = '1'
user_type = 'expert'
gaze_loss_type = 'KL'
checkpoint='43000'

#gpu = ['2','3']
gpu = ['0','1','2','3','4','5','6','7']
i=0


bash_file_name = 'eval_'+user_type+'_conv'+conv_layer+'_'+gaze_loss_type+'_'+checkpoint+'.sh'
#bash_file_name = 'eval_'+user_type+'_'+checkpoint+'.sh'
f = open(bash_file_name,'w')
f.write("#!/bin/bash\n")

for env in env_names:
  for seed in seeds:
    gpu_id = gpu[i%8]
    screen_name = 'eval_'+env[0]+'_'+gaze_loss_type+'_'+seed+'_'+checkpoint
    #f.write("#!/bin/bash")
    f.write("screen -dmS "+screen_name+" bash\n")
    f.write("screen -S "+screen_name+" -X stuff \"cd\n")
    f.write("\"\n")
    f.write("screen -S "+screen_name+" -X stuff \". ./setup_trex.sh\n")
    f.write("\"\n")
                  
    f.write("screen -S "+screen_name+" -X stuff \"python evaluateLearnedPolicy.py --env_name "+env[0]+" --checkpointpath path_to_logs/"+user_type+"s/"+env[0]+"_"+user_type+"_conv"+conv_layer+'_'+gaze_loss_type+'_'+seed+"/checkpoints/"+checkpoint+"\n")
    #f.write("screen -S "+screen_name+" -X stuff \"python evaluateLearnedPolicy.py --env_name "+env[0]+" --checkpointpath path_to_logs/"+env[0]+"_"+user_type+'_'+seed+"/checkpoints/"+checkpoint+"\n")
    f.write("\"\n")
    i+=1
      
f.close()
        
