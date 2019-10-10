import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import preprocess

from cnn import Net
import atari_head_dataset as ahd 
import utils

def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, gaze_coords, use_gaze):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    training_gaze = []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3,7)
        
        traj_i = demonstrations[ti][si::step]  #slice(start,stop,step)
        traj_j = demonstrations[tj][sj::step]
        
        if use_gaze:
            gaze_i = gaze_coords[ti][si::step]
            gaze_j = gaze_coords[tj][si::step]

        if ti > tj:
            label = 0
        else:
            label = 1
        
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        if use_gaze:
			training_gaze.append((gaze_i, gaze_j))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    # TODO: why 2 for loops?
    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        if use_gaze:
            gaze_i = gaze_coords[ti][ti_start:ti_start+rand_length:2]
            gaze_j = gaze_coords[tj][tj_start:tj_start+rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        if use_gaze:
			training_gaze.append((gaze_i, gaze_j))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, training_gaze



def get_gaze_coverage_loss(self, true_gaze, conv_gaze):
		loss = 0
		batch_size = true_gaze.shape[0]
        print('batch size: ', batch_size)

		# sum over all dimensions of the conv map
		conv_gaze = conv_gaze.sum(dim=1)
		# print(conv_gaze.shape)

		# collapse and normalize the conv map
		min_x = torch.min(torch.min(conv_gaze,dim=1)[0],dim=1)[0]
		max_x = torch.max(torch.max(conv_gaze,dim=1)[0], dim=1)[0]
		
		min_x = min_x.reshape(batch_size,1).repeat(1,7).unsqueeze(-1).expand(batch_size,7,7)
		max_x = max_x.reshape(batch_size,1).repeat(1,7).unsqueeze(-1).expand(batch_size,7,7)
		x_norm = (conv_gaze - min_x)/(max_x - min_x)

		# assert batch size for both conv and true gaze is the same
		assert(batch_size==conv_gaze.shape[0])
		
        # TODO:  batch size == 1?
		coverage_loss = torch.sum(torch.bmm(true_gaze,torch.abs(true_gaze-x_norm)))/batch_size

		return coverage_loss



# Train the network
def learn_reward(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_dir, gaze_loss_type, gaze_reg):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    # gaze loss
    if gaze_loss_type=='EMD':
		gaze_loss = gaze_loss_EMD
	elif gaze_loss_type=='coverage':
		gaze_loss = gaze_loss_coverage
	elif gaze_loss_type=='KL':
		gaze_loss = gaze_loss_KL

    loss_criterion = nn.CrossEntropyLoss()
    
    cum_loss = 0.0
    training_inputs, training_outputs, training_gaze7 = training_data
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            output_loss = loss_criterion(outputs, labels)

            if gaze_loss_type is None:
				loss = output_loss + l1_reg * abs_rewards
            elif gaze_loss_type=='coverage':
                # ground truth human gaze maps (7x7)
				gaze7_i, gaze7_j = training_gaze7[i]
                # TODO: gaze7_i, gaze7_j are tensors?
                print(type(gaze7_i))

                gaze_loss_i = gaze_loss(gaze7_i, conv_map_i)
				gaze_loss_j = gaze_loss(gaze7_j, conv_map_j)

                gaze_loss_total = (gaze_loss_i + gaze_loss_j)
                print('gaze loss: ', gaze_loss_total.data)    

                loss = output_loss + l1_reg * abs_rewards + gaze_reg * gaze_loss_total

            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 6000, type = int, help = "number of short subtrajectories to sample")

	parser.add_argument('--data_dir', help="where atari-head data is located")
	parser.add_argument('--gaze_loss', default=None, type=str, help="type of gaze loss function: EMD, coverage, KD, None")
	parser.add_argument('--gaze_reg', default=0.01, help="gaze loss multiplier")
	# parser.add_argument('--metric', default='rewards', help="metric to compare paired trajectories performance: rewards or returns")
	parser.add_argument('--gaze_dropout', default=False, action='store_true', help="use gaze modulated dropout or not")


    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    num_trajs =  args.num_trajs
    num_snippets = args.num_snippets
    min_snippet_length = 50 #min length of trajectory for training comparison
    maximum_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True

    # gaze-related arguments
    use_gaze = args.gaze_dropout or (args.gaze_loss is not None)
	gaze_loss_type = args.gaze_loss
	gaze_reg = float(args.gaze_reg)
	# mask = args.mask_scores
	gaze_dropout = args.gaze_dropout

    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    # demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)
    # Use Atari-HEAD human demos
    data_dir = args.data_dir
	dataset = ahd.AtariHeadDataset(env_name, data_dir)
	demonstrations, learning_returns, learning_rewards, learning_gaze7 = utils.get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze)


    #sort the demonstrations according to ground truth reward to simulate ranked demos

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    learning_gaze26 = [x for _, x in sorted(zip(learning_returns,learning_gaze26), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    
    training_data = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, learning_gaze7, use_gaze)
    training_obs, training_labels, training_gaze7 = training_data
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
   
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(gaze_dropout)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path, gaze_dropout)
    torch.cuda.empty_cache() 


    #save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels, training_gaze26, gaze_dropout))

    
