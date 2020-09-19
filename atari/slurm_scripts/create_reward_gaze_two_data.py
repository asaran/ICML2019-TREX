#env_names = ['asterix','breakout','centipede','phoenix','seaquest','mspacman']
env_names = ['alien','asterix','bank_heist','berzerk','breakout','centipede','demon_attack','enduro','freeway','frostbite','hero','montezuma_revenge','mspacman','name_this_game','phoneix','riverraid','road_runner','seaquest','space_invaders','venture']
seeds = ['0']
weights = ['0.001','0.005','0.01','0.05','0.1','0.3','0.5','0.7','0.9']

import os
for game in env_names:
    for seed in seeds:
        for weight in weights:
            d = 'reward_models_AAAI2020/gazePred_'+game+'_KL_'+weight+'_seed'+seed+'_two-data'
            #if not os.path.isdir('../'+d):
            #    os.mkdir('../'+d)

            #f.write('#SBATCH --job-name asterix_'+weight+'_'+seed)
            screen_name = 'two-data_'+game+'_seed'+seed+'_'+weight
            bash_file_name = 'gaze_pred/reward_'+game+'_human_kl_seed'+seed+'_'+weight+'two-data.sh'
            f = open(bash_file_name,'w')

            f.write("#!/bin/bash\n\n")
            f.write("#SBATCH --job-name "+screen_name+" \n")
            f.write("#SBATCH --output=logs/human_%j.out\n")
            f.write("#SBATCH --error=logs/human_%j.err\n")
            f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
            f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
            f.write("#SBATCH --partition dgx\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH --time 72:00:00\n")
            f.write("#SBATCH --gres=gpu:4\n")
            f.write("#SBATCH --mem=70G\n")
            f.write("#SBATCH --cpus-per-task=8\n")

            
            f.write("python LearnAtariRewardGaze.py --env_name "+game+" --data_dir ../../atari-head-two/ --gaze_loss KL --reward_model_path "+d+" --seed "+seed+" --gaze_reg "+weight)
            #f.write("python LearnAtariRewardGaze.py --env_name "+game+" --data_dir ../../atari-head/--reward_model_path "+d+" --seed "+seed)
