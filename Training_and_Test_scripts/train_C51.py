import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
from collections import deque



from c51 import *



gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": False,  # enable (debug) output of veins
    "port": 5558,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")


#env = gym.make("CartPole-v0") ### Instance of Veins gym 
#NUM_ACTIONS = env.action_space.n
NUM_ACTIONS = 8
#NUM_STATES = env.observation_space.shape[0]
NUM_STATES = 4


net, target_net = QNet(), QNet()
target_net.load_state_dict(net.state_dict())
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
buffer = deque(maxlen=buffer_size)
score, step = 0.0, 0
epsilon, epsilon_decay = 0.4, 1-5e-5
target_interval = 10

reward_list = []
    
for ep in tqdm.tqdm(range(EPISODES)):
    obs = env.reset()
    done = False
    score = 0
    while not done:
       acts_dist = net(torch.tensor(obs).unsqueeze(0).float())
       acts_val = np.array([expect(acts_dist[idx]).item() for idx in range(num_act)])
       rand = random.random()
       if rand < epsilon:
           action = random.randint(0, num_act-1)
       else:
           action = acts_val.argmax()
       next_obs, reward, done, info = env.step(action)
       buffer.append((obs, action, reward/50.0, next_obs, done))
       obs = next_obs
       step += 1
       score += reward
       epsilon *= epsilon_decay
            
    if len(buffer) > start_train:
       train(net, target_net, optimizer, buffer)
            
    if ep%target_interval==0 and ep!=0:
       target_net.load_state_dict(net.state_dict())
            
    if ep%10==0 and ep!=0:
       print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
             score/10.0, len(buffer), epsilon))
       
    reward_list.append(score)
	   

check_point = 'model_C51.pt'

net.save(check_point)       


np.save('C51_rewards.npy', reward_list)
#plt.plot(reward_list)
#plt.xlabel("Episode")
#plt.ylabel("Reward")
#plt.grid()
#plt.show()       

    
