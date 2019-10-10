import numpy as np
import cv2
import csv
import os
import torch
from os import path, listdir
import gaze_heatmap as gh
import time
from baselines.common.trex_utils import preprocess

cv2.ocl.setUseOpenCL(False)

def normalize_state(obs):
    return obs / 255.0

def normalize(obs, max_val):
    #TODO: discard frames with no gaze
    if(max_val!=0):
        norm_map = obs/float(max_val)
    else:
        norm_map = obs
    return norm_map

def mask_score(obs, crop_top = True):
    if crop_top:
        #takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        #no_score_obs = copy.deepcopy(obs)
        obs[:,:n,:,:] = 0
    else:
        n = 20
        obs[:,-n:,:,:] = 0
    return obs

#need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width=84
    height=84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    #frame = np.expand_dims(frame, -1)
    return frame

def MaxSkipAndWarpFrames(trajectory_dir, img_dirs, frames):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(frames)
    # print('total images:', num_frames)
    skip=4
    
    sample_pic = np.random.choice(listdir(path.join(trajectory_dir,img_dirs[0])))
    image_path = path.join(trajectory_dir, img_dirs[0], sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    for i in range(num_frames):
        #TODO: check that i should max before warping.
        img_name = frames[i] + ".png"
        img_dir =  img_dirs[i]

        if i % skip == skip - 2:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))
            
            obs_buffer[0] = obs
        if i % skip == skip - 1:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))
            obs_buffer[1] = obs
            
            #warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
    return max_frames

def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84,84,4))
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:,:,0] = frames[i-3]
            stacked_obs[:,:,1] = frames[i-2]
            stacked_obs[:,:,2] = frames[i-1]
            stacked_obs[:,:,3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
    return stacked


def CreateGazeMap(gaze_coords, pic):
    import math
    w, h = 7, 7
    old_h, old_w = pic.shape[0], pic.shape[1]
    obs = np.zeros((w, h))
    # print(gaze_coords)
    gaze_freq = {}
    if(not np.isnan(gaze_coords).all()):      
        for j in range(0,len(gaze_coords),2):
            if(not np.isnan(gaze_coords[j]) and not np.isnan(gaze_coords[j+1])):
                x = (gaze_coords[j])*w/old_w
                y = (gaze_coords[j+1])*h/old_h

                x, y = min(int(x),w-1), min(int(y),h-1)
                
                if (x,y) not in gaze_freq:
                    gaze_freq[(x,y)] = 1
                else:
                    gaze_freq[(x,y)] += 1
    
    # Create the gaze mask based on how frequently a coordinate is fixated upon
    for coords in gaze_freq:
        x, y = coords
        obs[y,x] = gaze_freq[coords]

    if np.isnan(obs).any():
        print('nan gaze map created')
        exit(1)

    return obs

def MaxSkipGaze(gaze, heatmap_size):
    """take a list of gaze coordinates and max over every 3rd and 4th observation"""
    num_frames = len(gaze)
    skip=4
    
    obs_buffer = np.zeros((2,)+(heatmap_size,heatmap_size), dtype=np.float32)
    max_frames = []
    for i in range(num_frames):
        g = gaze[i]
        g = np.squeeze(g)
        if i % skip == skip - 2:
            obs_buffer[0] = g
        if i % skip == skip - 1:
            obs_buffer[1] = g
            image = obs_buffer.max(axis=0)
            max_frames.append(image)
    if np.isnan(max_frames).any():
        print('nan max gaze map created')
        exit(1)
            
    return max_frames


def MaxGazeHeatmaps(gaze_hm_1, gaze_hm_2, heatmap_size):
    '''takes 2 stacks of 4 heatmaps for an observation (corresponding to 3rd and 4th frame) and take the max of the two'''
    obs_buffer = np.zeros((2,)+(heatmap_size,heatmap_size), dtype=np.float32)
    max_frames = []

    for gh_i,gh_j in zip(gaze_hm_1, gaze_hm_2):
        obs_buffer[0] = gh_i
        obs_buffer[1] = gh_j
        image = obs_buffer.max(axis=0)
        max_frames.append(image)

    return max_frames

def SkipGazeCoords(gaze_coords):
    """take a list of gaze coordinates and uses every 3rd and 4th observation"""
    num_frames = len(gaze_coords)
    skip=4
    
    skipped_frames = []
    obs_buffer = []
    for i in range(num_frames):
        g = gaze_coords[i]
        
        if i % skip == skip - 2:
            obs_buffer.append(g)
        if i % skip == skip - 1:
            obs_buffer.append(g)
            skipped_frames.append(obs_buffer)
            obs_buffer = []
    
    return skipped_frames

def CollapseGaze(gaze_frames, heatmap_size):
    import copy
    """combine every four frames to make an observation (84,84)"""
    stacked = []
    stacked_obs = np.zeros((heatmap_size,heatmap_size))
    for i in range(len(gaze_frames)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs = gaze_frames[i-3]
            stacked_obs = stacked_obs + gaze_frames[i-2]
            stacked_obs = stacked_obs + gaze_frames[i-1]
            stacked_obs = stacked_obs + gaze_frames[i]

            # Normalize the gaze mask
            max_gaze_freq = np.amax(stacked_obs)
            stacked_obs = normalize(stacked_obs, max_gaze_freq)

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)

    return stacked

def CollapseGazeHeatmaps(maxed_gaze, heatmap_size):
    import copy
    """combine four frames to make an observation (84,84)"""
    stacked_obs = np.zeros((heatmap_size,heatmap_size))
    for i in range(len(maxed_gaze)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs = maxed_gaze[i-3]
            stacked_obs = stacked_obs + maxed_gaze[i-2]
            stacked_obs = stacked_obs + maxed_gaze[i-1]
            stacked_obs = stacked_obs + maxed_gaze[i]

            # Normalize the gaze mask
            max_gaze_freq = np.amax(stacked_obs)
            stacked_obs = normalize(stacked_obs, max_gaze_freq)

    return np.expand_dims(copy.deepcopy(stacked_obs),0) #(1,7,7)

def StackGaze(gaze_frames, heatmap_size):
    import copy
    """combine every four frames to make an observation (84,84)"""
    stacked = []
    stacked_obs = np.zeros((heatmap_size,heatmap_size,4))
    for i in range(len(gaze_frames)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs[:,:,0] = gaze_frames[i-3]
            stacked_obs[:,:,1] = gaze_frames[i-2]
            stacked_obs[:,:,2] = gaze_frames[i-1]
            stacked_obs[:,:,3] = gaze_frames[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)

    return stacked

def StackGazeCoords(gaze_coord_pairs):
    import copy
    """combine every four coordinate pairs to make an observation"""
    stacked = []
    for i in range(len(gaze_coord_pairs)):
        stacked_obs = []
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs.append(gaze_coord_pairs[i-3])
            stacked_obs.append(gaze_coord_pairs[i-2])
            stacked_obs.append(gaze_coord_pairs[i-1])
            stacked_obs.append(gaze_coord_pairs[i])

            # stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)
            stacked.append(stacked_obs)

    return stacked

def MaxSkipReward(rewards):
    """take a list of rewards and max over every 3rd and 4th observation"""
    num_frames = len(rewards)
    skip=4
    max_frames = []
    obs_buffer = np.zeros((2,))
    for i in range(num_frames):
        r = rewards[i]
        if i % skip == skip - 2:
            
            obs_buffer[0] = r
        if i % skip == skip - 1:
            
            obs_buffer[1] = r
            rew = obs_buffer.max(axis=0)
            max_frames.append(rew)
    return max_frames


def StackReward(rewards):
    import copy
    """combine every four frames to make an observation"""
    stacked = []
    stacked_obs = np.zeros((1,))
    for i in range(len(rewards)):
        if i >= 3:
            # Sum over the rewards across four frames
            stacked_obs = rewards[i-3]
            stacked_obs = stacked_obs + rewards[i-2]
            stacked_obs = stacked_obs + rewards[i-1]
            stacked_obs = stacked_obs + rewards[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
    return stacked

def get_sorted_traj_indices(env_name, dataset):
    #need to pick out a subset of demonstrations based on desired performance
    #first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    #need to sort indices based on 'score'
    game = env_name
    #Note, I'm also going to try only keeping the full demonstrations that end in terminal
    traj_indices = []
    traj_scores = []
    traj_dirs = []
    traj_rewards = []
    traj_gaze = []
    traj_frames = []
    print('traj length: ',len(dataset.trajectories[game]))
    for t in dataset.trajectories[game]:
        traj_indices.append(t)
        traj_scores.append(dataset.trajectories[game][t][-1]['score'])
        # a separate img_dir defined for every frame of the trajectory as two different trials could comprise an episode
        traj_dirs.append([dataset.trajectories[game][t][i]['img_dir'] for i in range(len(dataset.trajectories[game][t]))])
        traj_rewards.append([dataset.trajectories[game][t][i]['reward'] for i in range(len(dataset.trajectories[game][t]))])
        traj_gaze.append([dataset.trajectories[game][t][i]['gaze_positions'] for i in range(len(dataset.trajectories[game][t]))])
        traj_frames.append([dataset.trajectories[game][t][i]['frame'] for i in range(len(dataset.trajectories[game][t]))])

    sorted_traj_indices = [x for _, x in sorted(zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)
    sorted_traj_dirs = [x for _, x in sorted(zip(traj_scores, traj_dirs), key=lambda pair: pair[0])]
    sorted_traj_rewards = [x for _, x in sorted(zip(traj_scores, traj_rewards), key=lambda pair: pair[0])]
    sorted_traj_gaze = [x for _, x in sorted(zip(traj_scores, traj_gaze), key=lambda pair: pair[0])]
    sorted_traj_frames = [x for _, x in sorted(zip(traj_scores, traj_frames), key=lambda pair: pair[0])]

    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))

    #so how do we want to get demos? how many do we have if we remove duplicates?
    seen_scores = set()
    non_duplicates = []
    for i,s,d,r,g,f in zip(sorted_traj_indices, sorted_traj_scores, sorted_traj_dirs, sorted_traj_rewards, sorted_traj_gaze, sorted_traj_frames):
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i,s,d,r,g,f))
    print("num non duplicate scores", len(seen_scores))
    if env_name == "spaceinvaders":
        start = 0
        skip = 3
    elif env_name == "revenge":
        start = 0
        skip = 1
    elif env_name == "qbert":
        start = 0
        skip = 3
    elif env_name == "mspacman":
        start = 0
        skip = 1
    else:   # TODO: confirm best logic for all games
        start = 0
        skip = 3
    num_demos = 12
    # demos = non_duplicates[start:num_demos*skip + start:skip] 
    demos = non_duplicates # don't skip any demos
    return demos


def get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked
    """
    #mspacman score is on the bottom of the screen
    if env_name == 'mspacman':
        crop_top = False
    else:
        crop_top = True

    demos = get_sorted_traj_indices(env_name, dataset)
    human_scores = []
    human_demos = []
    human_rewards = []
    #human_gaze = []
    human_gaze_26 = []
    # human_gaze_11 = []
    human_gaze_7 = []

    # img_frames = []
    print('len demos: ',len(demos))
    for indx, score, img_dir, rew, gaze, frame in demos:
        print(indx)
        # print(img_dir)
        human_scores.append(score)

        # traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        traj_dir = path.join(data_dir, env_name)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir, img_dir, frame)
        stacked_traj = StackFrames(maxed_traj)

        demo_norm_mask = []
        #normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            # if mask_scores:
            #     demo_norm_mask.append(mask_score(normalize_state(ob), crop_top))
            # else:
            #     demo_norm_mask.append(normalize_state(ob))  # currently not cropping

            demo_norm_mask.append(preprocess(ob, env_name)[0])
        human_demos.append(demo_norm_mask)

        # skip and stack reward
        maxed_reward = MaxSkipReward(rew)
        stacked_reward = StackReward(maxed_reward)      
        human_rewards.append(stacked_reward)

        if(use_gaze):
            # just return the gaze coordinates themselves
            # skipped_gaze = SkipGazeCoords(gaze)
            # stacked_gaze = StackGazeCoords(skipped_gaze)
            # human_gaze.append(stacked_gaze)

            # generate gaze heatmaps as per Ruohan's algorithm
            h = gh.DatasetWithHeatmap()
            # g_26 = h.createGazeHeatmap(gaze, 26)
            # g_11 = h.createGazeHeatmap(gaze, 11)
            g_7 = h.createGazeHeatmap(gaze, 7)

            # print('g type: ', type(g_11))

            # skip and stack gaze
            # maxed_gaze_26 = MaxSkipGaze(g_26, 26)
            # stacked_gaze_26 = CollapseGaze(maxed_gaze_26, 26)
            # human_gaze_26.append(stacked_gaze_26)

            # maxed_gaze_11 = MaxSkipGaze(g_11, 11)
            # stacked_gaze_11 = CollapseGaze(maxed_gaze_11, 11)
            # human_gaze_11.append(stacked_gaze_11)

            maxed_gaze_7 = MaxSkipGaze(g_7, 7)
            stacked_gaze_7 = CollapseGaze(maxed_gaze_7, 7)
            human_gaze_7.append(stacked_gaze_7)

            # print('maxed gaze type: ',type(maxed_gaze_11)) #list
            # print('stacked gaze type: ',type(stacked_gaze_11)) #list

    if(use_gaze):    
        print(len(human_demos[0]), len(human_rewards[0]), len(human_gaze_7[0]))
        print(len(human_demos), len(human_rewards), len(human_gaze_7))
    return human_demos, human_scores, human_rewards, human_gaze_7


def read_gaze_file(game_file):
    with open(game_file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    return lines

def get_gaze_heatmap(gaze_coords, heatmap_size):
    '''generate gaze heatmps of specific size for an entire trajectory'''
    print('generating gaze heatmap for trajectory of size: ', heatmap_size)
    human_gaze = []

    # for each observation in the trajectory
    for gaze in gaze_coords:
    # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
        h = gh.DatasetWithHeatmap()
        gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
        gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

        # skip and stack gaze
        maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)
        stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)
        human_gaze.append(stacked_gaze)

    return human_gaze


def get_all_gaze_heatmaps(gaze_coords, heatmap_size):
    '''generate gaze heatmps of specific size for an all trajectories'''
    print('generating gaze heatmap of size: ', heatmap_size)
    human_gaze = [[],[]]
    i = 0 
    # for each demo pair
    print('no of demo pairs: ',len(gaze_coords))
    for demo_gaze_coords in gaze_coords:
        print(i)
        i += 1
        # for each observation in the trajectory
        traj_i_gaze, traj_j_gaze = [], []
        traj_i, traj_j = demo_gaze_coords[0], demo_gaze_coords[1]

        for gaze in traj_i:
            # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
            h = gh.DatasetWithHeatmap()
            gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
            gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

            # skip and stack gaze
            maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)

            stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)

            traj_i_gaze.append(stacked_gaze)
        human_gaze[0].append(traj_i_gaze)

        start = time.time()
        for gaze in traj_j:
            # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
            h = gh.DatasetWithHeatmap()
            gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
            gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

            # skip and stack gaze
            maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)
            stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)
            traj_j_gaze.append(stacked_gaze)
        human_gaze[1].append(traj_j_gaze)
        end = time.time()
        print('time for 50 heatmaps: ', end-start)

    print('generated 10,000 gaze heatmp pais of size: ', heatmap_size)
    return human_gaze



def generate_novice_demos(env, env_name, agent, model_dir):
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)

    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards