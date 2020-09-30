from os import path,listdir
import os
import numpy as np
import cv2
import torch

def normalize_state(obs):
    return obs / 255.0


#custom masking function for covering up the score/life portions of atari games
def mask_score(obs, env_name):
    obs_copy = obs.copy()
    if env_name in ["space_invaders","breakout","pong","spaceinvaders"]:
        #takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        obs_copy[:,:n,:,:] = 0
    elif env_name in ["beamrider"]:
        n_top = 16
        n_bottom = 11
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["enduro","alien"]:
        n_top = 0
        n_bottom = 14
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["bank_heist"]:
        n_top = 0
        n_bottom = 13
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["centipede"]:
        n_top = 0
        n_bottom = 10
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["hero"]:
        n_top = 0
        n_bottom = 30
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "qbert":
        n_top = 12
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["seaquest"]:
        n_top = 12
        n_bottom = 16
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
        #cuts out divers and oxygen

    elif env_name in ["mspacman","name_this_game"]:
        n_bottom = 15 #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["berzerk"]:
        n_bottom = 11 #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["riverraid"]:
        n_bottom = 18 #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["videopinball"]:
        n_top = 15
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["montezuma_revenge", "phoenix", "venture","road_runner"]:
        n_top = 10
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["asterix","demon_attack","freeway"]:
        n_top = 10
        n_bottom = 10
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["frostbite"]:
        n_top = 12
        n_bottom = 10
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    else:
        print("NOT MASKING SCORE FOR GAME: " + env_name)
        pass
    return obs_copy

def preprocess(ob, env_name):
    return mask_score(normalize_state(ob), env_name)


# need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def MaxSkipAndWarpFrames(img_dir):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    frames = os.listdir(img_dir)
    num_frames = len(frames)
    skip = 4

    sample_pic = np.random.choice(listdir(img_dir))
    image_path = path.join(img_dir, sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []

    for i in range(num_frames):
        img_name = frames[i] 

        if i % skip == skip - 2:
            obs = cv2.imread(path.join(img_dir, img_name))

            obs_buffer[0] = obs
        if i % skip == skip - 1:
            obs = cv2.imread(path.join(img_dir, img_name))
            obs_buffer[1] = obs

            # warp max to 84x84 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)

    return max_frames

def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84, 84, 4))
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:, :, 0] = frames[i-3]
            stacked_obs[:, :, 1] = frames[i-2]
            stacked_obs[:, :, 2] = frames[i-1]
            stacked_obs[:, :, 3] = frames[i]
            stacked.append(copy.deepcopy(stacked_obs))
            
    return torch.tensor(stacked)

def create_test_data(data_dir, env_name):
    # load sample img stack to test the model
    traj_dir = path.join(data_dir, env_name)
    maxed_traj= MaxSkipAndWarpFrames(data_dir)
    stacked_traj = StackFrames(maxed_traj)
    # print(stacked_traj.shape)

    # normalizing images
    demo_norm = normalize_state(stacked_traj)

    return demo_norm