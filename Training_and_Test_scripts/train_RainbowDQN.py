from Agents.RainbowDQN import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import argparse
import tqdm 

parser = argparse.ArgumentParser()
    
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--prior_eps', type=float)
parser.add_argument('--atom_size', type=int)
parser.add_argument('--namefile', type=str)
args = parser.parse_args()

gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": False,  # enable (debug) output of veins
    "port": 5560,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")


#env = gym.make("CartPole-v0") ### Instance of Veins gym 
#NUM_ACTIONS = env.action_space.n
NUM_ACTIONS = 8
#NUM_STATES = env.observation_space.shape[0]
NUM_STATES = 4

memory_size = 10000
batch_size = 128
target_update = 100
#dqn = RainbowDQNAgent(env = env, num_states = NUM_STATES, num_actions = NUM_ACTIONS, memory_size = memory_size, batch_size = batch_size, target_update = target_update)
dqn = RainbowDQNAgent(env = env, num_states = NUM_STATES, num_actions = NUM_ACTIONS, memory_size = memory_size, batch_size = batch_size, target_update = target_update, alpha = args.alpha, beta = args.beta, prior_eps = args.prior_eps, atom_size = args.atom_size)
episodes = 2000
horizon = 3680
reward_list = []
#plt.ion()
# fig, ax = plt.subplots()
for i in tqdm.tqdm(range(1, episodes+1)):
    t = 1
    state = env.reset()
    ep_reward = 0
    update_cnt = 0
    while True:
		
		
        action = dqn.select_action(state)
        #print(action)
        next_state, reward, done = dqn.step(action.item())


        state = next_state
        ep_reward += reward
           
        # NoisyNet: removed decrease of epsilon
           
        # PER: increase beta
        fraction = min(t / episodes*horizon, 1.0)
        dqn.beta = dqn.beta + fraction * (1.0 - dqn.beta)
        t = t +1
           # if episode ends
        if done:
           reward_list.append(ep_reward)
           break
        
           # if training is ready
        if len(dqn.memory) >= dqn.batch_size:
           loss = dqn.update_model()
           #losses.append(loss)
           update_cnt += 1
               
           # if hard update is needed
        if update_cnt % dqn.target_update == 0:
           dqn._target_hard_update()

       
#check_point = 'model_rainbow_dqn.pt'

#dqn.save(check_point)    
#dqn.save(args.namefile)   
path = args.namefile  
np.save(path, reward_list)
#plt.plot(reward_list)
#plt.xlabel("Episode")
#plt.ylabel("Reward")
#plt.grid()
#plt.show()
 

