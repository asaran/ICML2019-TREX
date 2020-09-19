env_names = ['asterix','breakout','centipede','phoenix','seaquest','mspacman']
seeds = ['0']
weights = ['0.001','0.005','0.01','0.05','0.1','0.3','0.5','0.7','0.9']

import os
for game in env_names:
    for seed in seeds:
        for weight in weights:
            d = 'reward_models_AAAI2020/'+game+'_seed'+'_'+seed
            
            #if not os.path.isdir('../'+d):
            #    os.mkdir('../'+d)

            #f.write('#SBATCH --job-name asterix_'+weight+'_'+seed)
            screen_name = game+'_seed'+seed
            bash_file_name = 'gaze_pred/reward_'+game+'_human_seed'+seed+'.sh'
            f = open(bash_file_name,'w')

            f.write("#!/bin/bash\n\n")
            f.write("#SBATCH --job-name "+screen_name+" \n")
            f.write("#SBATCH --output=logs/human_%j.out\n")
            f.write("#SBATCH --error=logs/human_%j.err\n")
            f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
            f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
            f.write("###SBATCH --partition Test\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH --time 72:00:00\n")
            f.write("#SBATCH --gres=gpu:4\n")
            f.write("#SBATCH --mem=70G\n")
            f.write("#SBATCH --cpus-per-task=8\n")

            
            #f.write("python LearnAtariRewardGaze.py --env_name "+game+" --data_dir ../../atari-head/ --gaze_loss KL --reward_model_path "+d+" --seed "+seed+" --gaze_reg "+weight)
            f.write("python LearnAtariRewardGaze.py --env_name "+game+" --data_dir ../../atari-head/ --reward_model_path "+d+" --seed "+seed)
