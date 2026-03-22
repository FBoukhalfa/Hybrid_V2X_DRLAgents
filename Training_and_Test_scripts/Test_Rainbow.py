## ======================== Testing the learned strategy =========================================
from RainbowDQN import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

def test(env,path):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    

    max_ep_len = 3680                   # max timesteps in one episode
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    # state space dimension
    #state_dim = env.observation_space.shape[0]
    state_dim = 4
    # action space dimension
    
    #action_dim = env.action_space.shape[0]
    #action_dim = env.action_space.n
    action_dim = 8
    ################# training procedure ################
    #has_continuous_action_space = True
    has_continuous_action_space = False

    #NUM_ACTIONS = env.action_space.n
    NUM_ACTIONS = 8
    #NUM_STATES = env.observation_space.shape[0]
    NUM_STATES = 4

    memory_size = 10000
    batch_size = 128
    target_update = 100
    param = path.split("_")
    alpha = float(param[1])
    beta = float(param[2])
    prior_eps = float(param[3])
    atom_size = int(param[4].split(".")[0])
    Rainbow_agent = RainbowDQNAgent(env = env, num_states = NUM_STATES, num_actions = NUM_ACTIONS, memory_size = memory_size, batch_size = batch_size, target_update = target_update, gamma = gamma, alpha = alpha, beta = beta, prior_eps = prior_eps, v_min= 0, v_max = 200, atom_size = atom_size)
    Rainbow_agent.load(path)  # We load the learned optimal policy that has been saved in the file model.pt
    
    # track total training time
    
    # logging file
    

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    episodic_return = []
   
 

    state = env.reset()

       
    done = False
    #for t in range(1, max_ep_len+1):
    while not done:
        #print("...")
        # select action with policy
        action = Rainbow_agent.select_action(state)
        # # Take the action and get the new observation space
        state, reward, done, _ = env.step(action.item())
        print(reward)


gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
    "scenario_dir": "../scenario",
    "print_veins_stdout": True,  # enable (debug) output of veins
    "run_veins": True,  # do not start veins through Veins-Gym
    "port": 5558,  # pick a port to use
    #"user_interface": "Qtenv",
    "timeout": 10,
    },
)

env = gym.make("veins-v1")

# To load the learned model, use the following command:
# initialize a PPO agent


    
# Instantiate an environment  --- > env
env.reset()

# path = "model_rainbow_dqn.pt"
# path = "logTrainRainbow_0.1_0.5_0.0001_25.pt" # 94.02%
# path = "logTrainRainbow_0.1_0.5_0.0001_100.pt" # 97.02% 
# path = "logTrainRainbow_0.1_0.5_0.0001_25.pt" # 94.000%
# path = "logTrainRainbow_0.1_0.5_1e-05_25.pt" # 95.85%
# path = "logTrainRainbow_0.1_0.7_0.0001_25.pt" # 96.24%
# path = "logTrainRainbow_0.1_0.7_1e-05_100.pt" # 81.358%
# path = "logTrainRainbow_0.1_0.7_1e-05_25.pt" # 97.28%
# path = "logTrainRainbow_0.3_0.5_0.0001_100.pt" # 97.35%
# path = "logTrainRainbow_0.3_0.5_0.0001_25.pt" # 96.36%
# path = "logTrainRainbow_0.3_0.5_1e-05_25.pt" # 96.90%
path = "logTrainRainbow_0.3_0.7_0.0001_25.pt" # 97.57%
# path = "logTrainRainbow_0.3_0.7_1e-05_100.pt" # 96.65%
# path = "logTrainRainbow_0.3_0.7_1e-05_25.pt" # 96.28%
# path = "logTrainRainbow_0.1_0.7_0.0001_100.pt" # 96.23%
# path =  "logTrainRainbow_0.3_0.7_0.0001_100.pt" # 96.86%
# path =  "logTrainRainbow_0.3_0.5_0.00001_100.pt" # 74.80 %

test(env,path)
















