## ======================== Testing the learned strategy =========================================
import os
import glob
import time
from datetime import datetime
#from numba import jit, cuda
import torch
import numpy as np
import pickle as pkl
import gym
from Agents.DQN import DQN
import veins_gym
from veins_gym import veinsgym_pb2
import os
import argparse

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


    BATCH_SIZE = 128    
    LR = 0.01
    GAMMA = 0.99
    EPISILO = 0
    MEMORY_CAPACITY = 2000
    Q_NETWORK_ITERATION = 100
  
    dqn = DQN(env = env, BATCH_SIZE = BATCH_SIZE, LR=LR, GAMMA = GAMMA, EPISILO = EPISILO, MEMORY_CAPACITY = MEMORY_CAPACITY, Q_NETWORK_ITERATION = Q_NETWORK_ITERATION)

    dqn.load(path)  # We load the learned optimal policy that has been saved in the file model.pt
    
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
        action = dqn.choose_action(state, True)
        # # Take the action and get the new observation space
        state, reward, done, _ = env.step(action)
        print(reward)


gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
    "scenario_dir": "../scenario",
    "print_veins_stdout": True,  # enable (debug) output of veins
    "run_veins": True,  # do not start veins through Veins-Gym
    "port": 5555,  # pick a port to use
    #"user_interface": "Qtenv",
    #"timeout": 10,
    },
)

env = gym.make("veins-v1")

# To load the learned model, use the following command:
# initialize a PPO agent


    
# Instantiate an environment  --- > env
env.reset()

path = "model_dqn.pt"
test(env,path)














