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
# from baselines.common.trex_utils import preprocess

from cnn import Net
# import atari_head_dataset as ahd 
import utils
from tensorboardX import SummaryWriter

from gaze import gaze_heatmap as gh
from gaze import human_utils
from gaze import atari_head_dataset as ahd
from baselines.common.trex_utils import preprocess, mask_score
from gaze.coverage import *


def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, use_gaze, use_human_gaze=False):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    training_gaze = []
    num_demos = len(demonstrations)

    if use_human_gaze:
        demos, learning_returns, _, learning_gaze = human_utils.get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze=True)
    else:
        demos, learning_returns, _ = human_utils.get_preprocessed_trajectories(env_name, dataset, data_dir)
    
    if use_gaze and not use_human_gaze:
        # get gaze prediction
        model_path = './gaze/pretrained_models/expert/' \
            + env_name + '.hdf5'
        meanfile_path = './gaze/pretrained_models/means/' + env_name + '.mean.npy'
        h = gh.PretrainedHeatmap(env_name, model_path, meanfile_path)
        heatmap_shape = 84


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
        
        # if use_gaze:
        #     gaze_i = gaze_coords[ti][si::step]
        #     gaze_j = gaze_coords[tj][sj::step]

        # TODO: add gaze prediction for enduro
        
        if ti > tj:
            label = 0
        else:
            label = 1
        
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        if use_gaze:
            training_gaze.append((gaze_i, gaze_j))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

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
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip every other framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        # if use_gaze:
        #     gaze_i = gaze_coords[ti][ti_start:ti_start+rand_length:2]
        #     gaze_j = gaze_coords[tj][tj_start:tj_start+rand_length:2]

        gaze_i, gaze_j = [], []
        if use_gaze and not use_human_gaze:

            # gaze = h.get_heatmap(ob, heatmap_shape) # already preprocessed (normalized and masked)
            gaze_i = h.get_heatmap(traj_i, heatmap_shape) 
            gaze_j = h.get_heatmap(traj_j, heatmap_shape) 
            # traj.append((ob_processed,gaze))
            # print('verifying: ', ob_processed.shape, action, gaze.shape) (1, 4, 84, 84) 0 (1, 84, 84)
        elif use_gaze and use_human_gaze:
            # gaze = learning_gaze[i][k]
            # traj.append((ob_processed,gaze))
            # gaze_i = gaze_coords[ti][ti_start:ti_start+rand_length:2]
            # gaze_j = gaze_coords[tj][tj_start:tj_start+rand_length:2]
            print('not implemented yet')

        # else:
            # traj.append((ob_processed))



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




# Train the network
def learn_reward(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_dir, gaze_loss_type, gaze_reg, gaze_conv_layer):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    
    import os
    from pathlib import Path
    if not os.path.isdir(checkpoint_dir+'_tb'):
        os.makedirs(checkpoint_dir+'_tb')
    writer = SummaryWriter(checkpoint_dir+'_tb')
    
    loss_criterion = nn.CrossEntropyLoss()
    cum_loss = 0.0
 
    training_inputs, training_outputs, training_gaze = training_data

    if gaze_loss_type in ['KL', 'IG']:
        training_data = list(zip(training_inputs, training_outputs, training_gaze))
    else:
        training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        if gaze_loss_type in ['KL', 'IG']:
            training_obs, training_labels, training_gaze = zip(*training_data)
        else:
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
            if gaze_loss_type in ['KL', 'IG']:
                outputs, abs_rewards, attn_map_i, attn_map_j = reward_network.forward(traj_i, traj_j, gaze_conv_layer)
            else:
                outputs, abs_rewards, _, _ = reward_network.forward(traj_i, traj_j)	
            outputs = outputs.unsqueeze(0)
            output_loss = loss_criterion(outputs, labels)

            loss = output_loss + l1_reg * abs_rewards
            # TODO: plot validation losses instead of training losses
            writer.add_scalar('CE_loss', loss.item(), epoch*len(training_labels)+i)

            if gaze_loss_type in ['KL', 'IG']:
                # ground truth human gaze maps (7x7)
                gaze_i, gaze_j = training_gaze[i]
                gaze_i = torch.squeeze(torch.tensor(gaze_i, device=device)) # list of torch tensors
                gaze_j = torch.squeeze(torch.tensor(gaze_j, device=device))


                if gaze_loss_type == 'KL':
                    eps = 0.0001
                    # N=84*84
                    # gaze_loss_i = get_gaze_KL_loss(gaze_i, torch.squeeze(conv_map_i))
                    # gaze_loss_j = get_gaze_KL_loss(gaze_j, torch.squeeze(conv_map_j))


                    
                    # attn_map_i = norm_operator(torch.squeeze(conv_map_i))
                    # attn_map_j = norm_operator(torch.squeeze(conv_map_j))
                    # print(attn_map.shape)
                    # attn_shape = torch.Size([attn_map.shape[0], attn_map.shape[1], 84,84])
                    attn_map_i = F.interpolate(attn_map_i, (84,84), mode="bilinear", align_corners=False)
                    attn_map_j = F.interpolate(attn_map_j, (84,84), mode="bilinear", align_corners=False)

                    kl_i = computeKL_batch(attn_map_i, gaze_i)
                    kl_j = computeKL_batch(attn_map_j, gaze_j)
                    gaze_loss_i = torch.sum(kl_i)
                    gaze_loss_j = torch.sum(kl_j)

                    gaze_loss_total = (gaze_loss_i + gaze_loss_j)
                    # TODO: use weighted loss for plotting (gaze_reg*gaze_loss_total)
                    writer.add_scalar('KL_loss', gaze_loss_total.item(), epoch*len(training_labels)+i) 

                loss = (1-gaze_reg)*loss + gaze_reg * gaze_loss_total
                writer.add_scalar('total_loss', loss.item(), epoch*len(training_labels)+i)

            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")
    writer.close()


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
            outputs, abs_return, _, _ = reward_network.forward(traj_i, traj_j)
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
            print('output of net.cum_return: ',r)
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
    parser.add_argument('--gaze_loss', default=None, type=str, help="type of gaze loss function: sinkhorn, exact, coverage, KL, None")
    parser.add_argument('--gaze_reg', default=0.01, type=float, help="gaze loss multiplier")
    parser.add_argument('--gaze_conv_layer', default=4, type=int, help='the convlayer of the reward network to which gaze should be compared')


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
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    num_trajs = args.num_trajs
    num_snippets = args.num_snippets
    min_snippet_length = 50 #min length of trajectory for training comparison
    maximum_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg = 0.0
    stochastic = True

    # gaze-related arguments
    use_gaze = args.gaze_loss
    gaze_loss_type = args.gaze_loss
    gaze_reg = args.gaze_reg
    gaze_conv_layer = args.gaze_conv_layer
    print('*************** GAZE: ',use_gaze,'****************')

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

    print('preparing data..')
    demonstrations, learning_returns, learning_rewards = utils.get_preprocessed_trajectories(env_name, dataset, data_dir)

    # if use_motion:
    #     use_gaze=True
    #     gaze_loss_type = args.motion_loss

    #sort the demonstrations according to ground truth reward to simulate ranked demos
    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    # learning_gaze = [x for _, x in sorted(zip(learning_returns,learning_gaze), key=lambda pair: pair[0])]
    sorted_returns = sorted(learning_returns)
    
    print('collecting traj snippets...')
    training_data = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, use_gaze)
    training_obs, training_labels, training_gaze = training_data

    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
   
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(gaze_loss_type)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    print('learning reward function...')
    learn_reward(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path, gaze_loss_type, gaze_reg, gaze_conv_layer)
    torch.cuda.empty_cache() 

    #save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))