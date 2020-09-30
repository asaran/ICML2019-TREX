import torch
import numpy as np
import cv2
from trex_utils import create_test_data
import argparse
from cnn import Net

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='asterix', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='models/asterix_seed_0', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--data_dir', default='data/asterix/260_RZ_1456515_Mar-01-10-10-36/', help="test images")

    args = parser.parse_args()
    model_path = args.reward_model_path

    # check if GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(None)
    model_dict = torch.load(model_path, map_location=device)
    reward_net.load_state_dict(model_dict)
    
    # load data
    data = create_test_data(args.data_dir, args.env_name)
    batch_size = data.shape[0]
    print('Processing %d frame stacks',batch_size)

    # forward pass on loaded model
    reward = reward_net.single_forward(data.float())
    print('reward tensor shape: ', reward.shape)
    print('reward: ', reward)