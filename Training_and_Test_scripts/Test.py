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
from PPO import PPO
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
  
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    ppo_agent.load(path)  # We load the learned optimal policy that has been saved in the file model.pt
    
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
        action = ppo_agent.select_action(state, True)
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
#parser = argparse.ArgumentParser()
#parser.add_argument('--namefile', type=str)
#args = parser.parse_args()
# Reconstruct the actorCritic based on the learned model
# ppo_agent = PPO (model learned)
# path = args.namefile
# path = "logTrain_0.0001_0.0001_0.1.pt" # 95.351961311%
# path = "logTrain_0.001_1e-05_0.3.pt" # 67.20%
# path = "logTrain_0.001_0.001_0.3.pt"  # 94.208893485%
# path = "logTrain_1e-05_0.0001_0.2.pt" # 91.348737238%
# path = "logTrain_1e-05_0.0001_0.1.pt" # 91.348737238%
# path = "logTrain_1e-05_1e-05_0.2.pt" # 91.483073616%
# path = "logTrain_0.001_0.001_0.1.pt" # 97.286405159%
# path = "logTrain_0.001_0.001_0.2.pt" # 97.155126141%
# path = "logTrain_0.0001_0.001_0.1.pt" # 95.72810317%
# path = "logTrain_0.001_1e-05_0.2.pt" # 81.816112881%
# path = "logTrain_0.0001_1e-05_0.2.pt" # 93.415529906%
# path = "logTrain_1e-05_1e-05_0.3.pt" # 92.26222461%
# path = "logTrain_1e-05_0.001_0.1.pt" # 91.348737238%
# path = "logTrain_0.0001_1e-05_0.3.pt" # 93.715487773%
# path = "logTrain_0.0001_0.001_0.2.pt" # 94.171940928%
# path = "logTrain_0.0001_1e-05_0.1.pt" # 92.725383216%
# path = "logTrain_0.0001_0.001_0.3.pt" # 95.674368619%
# path = "logTrain_0.001_0.0001_0.2.pt" # 87.206516531%
# path = "logTrain_1e-05_0.001_0.3.pt" # 91.348737238%
# path = "logTrain_1e-05_0.0001_0.3.pt" # 91.348737238%
# path = "logTrain_0.001_0.0001_0.1.pt" # 92.888206702%
# path = "logTrain_1e-05_0.001_0.2.pt" # 92.26222461%
# path = "logTrain_0.01_0.01_0.1.pt" # 87.380038388%
# path = "logTrain_0.01_0.001_0.1.pt" # 84.225156359%
# path = "logTrain_0.01_0.001_0.01.pt" # 98.495836691%
# path = "logTrain_0.001_0.01_0.01.pt" # 95.862439549%
# path = "logTrain_0.001_0.001_0.1.pt" # 96.789727127%
# path = "logTrain_0.001_0.001_0.01.pt" # 94.825286743%
# path = "logTrain_0.01_0.01_0.01.pt"# 97.352454495%
# path = "logTrain_0.001_0.001_0.05.pt" # 96.707708779%
# path = "logTrain_0.01_0.001_0.05.pt" # 94.593895499%
# path = "logTrain_0.01_0.001_0.001.pt" # 91.396187532%
# path = "logTrain_0.01_0.01_0.05.pt" # 95.309665717%
# path = "logTrain_0.0001_0.0001_0.2.pt" # 94.545943041%
# path = "logTrain_0.0001_0.0001_0.3.pt" #  95.009390931%
# path = "logTrain_0.001_0.0001_0.3.pt" # 77.150138034%
# path = "logTrain_0.001_0.001_0.001.pt" # 94.96345515%
# path = "logTrain_0.001_0.01_0.001.pt" # 49.422353573%
# path = "logTrain_0.01_0.01_0.001.pt" # 46.77504626%
# path = "logTrain_1e-05_1e-05_0.1.pt" # 91.429339065%
# path = "logTrain_0.001_0.01_0.05.pt" # 97.716894977
# path = "logTrain_0.01_0.001_0.02.pt" # 94.57%
path = "logTrain_0.01_0.01_0.02.pt" # 97.57%s
# path = "logTrain_0.01_0.01_0.03.pt" # 97.208931419%
#path = "logTrain_0.01_0.001_0.01.pt" # 98.495836691%


test(env,path)


# path = "logTrain_0.001_0.001_0.2.pt" # 97.155126141%
#path = "logTrain_0.01_0.01_0.03.pt"  # 97.208931419%
# path = "logTrain_0.001_0.001_0.1.pt" # 97.286405159%
# path = "logTrain_0.01_0.01_0.01.pt"  # 97.352454495%
# path = "logTrain_0.01_0.01_0.02.pt"  # 97.57%
# path = "logTrain_0.01_0.001_0.01.pt" # 98.495836691%











