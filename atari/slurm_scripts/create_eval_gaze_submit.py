env_names = [['asterix','Asterix'],['centipede','Centipede'],['phoenix','Phoenix'],['breakout','Breakout'],['seaquest','Seaquest'],['mspacman','MsPacman']]

seeds = ['0']

gaze_loss_type = 'KL'
gaze_reg = ['0.001','0.005','0.01','0.05','0.1','0.3','0.5','0.7','0.9']

i=0

for env in env_names:
    for weight in gaze_reg:
        for seed in seeds:
            screen_name = 'eval_'+env[0]+'_'+gaze_loss_type+'_seed'+seed+'_'+weight

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
            f.write("#SBATCH --time 72:00:00\n") 
            f.write("#SBATCH --gres=gpu:1\n")
            f.write("#SBATCH --mem=50G\n")
            f.write("#SBATCH --cpus-per-task=8\n")
             
            f.write("python evaluateLearnedPolicy.py --env_name "+env[0]+" --checkpointpath "+"ppo_models/"+gaze_loss_type+"/"+env[0]+"_"+gaze_loss_type+"_"+weight+"_seed"+seed+"/checkpoints/43000")
            i+=1
      
            f.close()
        
