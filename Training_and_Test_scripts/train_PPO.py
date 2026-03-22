import os
import glob
import time
from datetime import datetime
#from num2 import jit, cuda
import torch
import numpy as np
import pickle as pkl
import gym
import argparse
import tqdm 

from Agents.PPO.PPO import PPO
import subprocess
################################### Training ###################################
#@jit(target_backend='cuda') 
parser = argparse.ArgumentParser()
    
parser.add_argument('--lr_actor', type=float)
parser.add_argument('--lr_critic', type=float)
parser.add_argument('--eps_clip', type=float)
parser.add_argument('--namefile', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()


def train(env):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    

    
    max_ep_len = 3680                   # max timesteps in one episode
    max_training_timesteps = 2000*max_ep_len #700*max_ep_len    # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
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

    #eps_clip = 0.3          # clip parameter for PPO # 0.2 -> smooth learning of dist #0.1 ==> pdr = 96.43% #0.3 ==> 96.67896679
    gamma = 0.99            # discount factor 

    #lr_actor = 0.001       # learning rate for actor network ==> 0.0003 ==> pdr = 94.000%
    #lr_critic = 0.001       # learning rate for critic network

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
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, gamma, K_epochs, args.eps_clip, has_continuous_action_space, action_std)

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
   
    # training loop
    
    while i_episode <= max_training_timesteps//max_ep_len:
        print(time_step)
        # Go to the next episode
        #os.system("./run -u Cmdenv --*.gym-connection.port=5555 > test_log.txt&")
     
        state = env.reset()

        

        current_ep_reward = 0
        done = False
        #for t in range(1, max_ep_len+1):
        while not done:
            #print("...")
            # select action with policy
            action = ppo_agent.select_action(state, False)
            # # Take the action and get the new observation space
            state, reward, done, _ = env.step(action)
            #print(reward)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()   # Learning 
                print("Updating the PPO Model")

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            

            # break; if the episode is over
            

        print("End episode")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        # Start the second episodes
        i_episode += 1

        episodic_return.append(current_ep_reward)


     
    #filename = 'PPO_returns.pkl'
    #fileObject = open(filename, 'wb') 
    #pkl.dump(np.array(episodic_return), fileObject)  
    #np.save('PPO_returns_2', np.array(episodic_return))
    #return ppo_agent

    path = args.namefile   
    #np.save('PPO_returns.npy', np.array(episodic_return))
    np.save(path, np.array(episodic_return))
    return ppo_agent



#from casadi import *


#import matplotlib.pyplot as plt
import numpy as np
#import tqdm




import gym
import veins_gym
import torch



gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": False,  # enable (debug) output of veins
    "port": 5555,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")

import os

#env = gym.make('Acrobot-v1')


#env.reset()   

#
#print(env.observation_space.shape[0])
#print(env.observation_space.sample())
#print(env.action_space.sample())


#print("Launch training")

print(env)
agent = train(env) # Training 

# At the end of the training phase, we assume that the agent has computed the optimal policy
#path = args.namefile
#path = "model.pt"

agent.save(model.pt) # Save the leanred optimal policy in the file untiled model.pt


#================  Observable stats ===============================

# Link the reward to the PDR




 ## ======================== Testing the learned strategy =========================================


# To load the learned model, use the following command:
#model = torch.load(path)


    
# Instantiate an environment  --- > env
#env.reset()
# Reconstruct the actorCritic based on the learned model
# ppo_agent = PPO (model learned)
# ppo_agent = PPO (model)
# while not done:
#    action = ppo_agent.select_action(state)    
#    state, reward, done, _ = env.step(action)
#    pdr = Fct_PDR(reward)
    
