import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--all_games", action="store_true", default=False)
parser.add_argument("--ckpt", type=int, default=43000)
parser.add_argument("--ppo_dir", type=str, default="ppo_models/base")

args = parser.parse_args()

if args.all_games:
     env_names = ['alien','asterix','bank_heist','berzerk','breakout','centipede','demon_attack','enduro','freeway','frostbite','hero','montezuma_revenge','mspacman','name_this_game','phoneix','riverraid','road_runner','seaquest','space_invaders','venture']
else:
     env_names = ['asterix','centipede','phoenix','breakout','seaquest','mspacman']

seeds = ['0']

dir_string = args.ppo_dir.split('/')

#gaze_loss_type = 'KL'
#gaze_reg = ['0.001','0.005','0.01','0.05','0.1','0.3','0.5','0.7','0.9']

i=0

for env in env_names:
    for seed in seeds:
        screen_name = 'eval_'+env+'_'+dir_string[1]+'_seed'+seed+'_'+str(args.ckpt)

        bash_file_name = 'gaze_pred/'+screen_name+'.sh'
        f = open(bash_file_name,'w')

        f.write("#!/bin/bash\n\n")
        f.write("#SBATCH --job-name "+screen_name+" \n")
        f.write("#SBATCH --output=logs/slurmjob_%j.out\n")
        f.write("#SBATCH --error=logs/slurmjob_%j.err\n")
        f.write("#SBATCH --mail-user=asaran@cs.utexas.edu\n")
        f.write("#SBATCH --mail-type=END,FAIL,REQUEUE\n")
        f.write("###SBATCH --partition titans\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        f.write("#SBATCH --time 72:00:00\n") 
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --mem=50G\n")
        f.write("#SBATCH --cpus-per-task=8\n")
             
        f.write("python evaluateLearnedPolicy.py --env_name "+env+" --checkpointpath "+args.ppo_dir+'/'+env+"_seed"+seed+"/checkpoints/"+str(args.ckpt))
        i+=1
      
        f.close()
        
