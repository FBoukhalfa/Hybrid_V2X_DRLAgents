import gym
#import pybullet_envs
import numpy as np
from collections import deque
import torch
import argparse
from Agents.SAC.buffer import ReplayBuffer
import glob
from Agents.SAC.utils import collect_random
import random
from Agents.SAC.SAC import SAC
import matplotlib.pyplot as plt 
import tqdm 

parser = argparse.ArgumentParser()
    
parser.add_argument('--tau', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--namefile', type=str)
args = parser.parse_args()
  



episodes = 2000
buffer_size =  100000
#env = 'CartPole-v0'


#--------------------------------------------------------------

gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": False,  # enable (debug) output of veins
    "port": 5559,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")
    
#env = gym.make('CartPole-v0')

state_dim = 4
action_dim = 8


#------------------- Parameters to change    -------------------------------------

# tau=1e-2   # 0.005
# learning_rate = 5e-4    #0.0003
# batch_size =  256   # 128 # 64
#---------------------------------------------------------------

target_entropy = None
    
#env = gym.make(env)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
steps = 0
average10 = deque(maxlen=10)
total_steps = 0
    
        
agent = SAC(state_size=state_dim, action_size=action_dim, device=device, tau=args.tau, learning_rate = args.learning_rate)

buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=args.batch_size, device=device)
        
collect_random(env=env, dataset=buffer, num_samples=10000)
        
reward_episodes = []
    
for i in tqdm.tqdm(range(1, episodes+1)):
    state = env.reset()
    episode_steps = 0
    rewards = 0
    while True:
        action = agent.get_action(state)
        steps += 1
        action = action.item()
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
        state = next_state
        rewards += reward
        episode_steps += 1
        if done:
            reward_episodes.append(rewards)
            break

            

    average10.append(rewards)
    total_steps += episode_steps
    #print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))

path = args.namefile		
np.save(path, reward_episodes)    

# np.save('SAC_rewards.npy', reward_episodes)    
# plt.plot(reward_episodes)
	
#check_point = 'model_SAC.pt'

#path = args.namefile

#agent.save(path) 