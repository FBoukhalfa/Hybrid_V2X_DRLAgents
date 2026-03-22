from scenario.Delivrable_Github.Agent.SAC.SAC import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

#------------------- Parameters to change    -------------------------------------

tau=1e-2   # 0.005
learning_rate = 5e-4    #0.0003
batch_size =  256   # 128 # 64
#---------------------------------------------------------------




gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": True,  # enable (debug) output of veins
    "port": 5558,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")
   
# path = 'model_SAC.pt' 
# path = 'logTrain_SAC_0.005_0.0003_128.pt' # 97.02%
# path = 'logTrain_SAC_0.005_0.0003_256.pt' # 94.53%
# path = 'logTrain_SAC_0.005_0.0003_64.pt' # 95.51%
# path = 'logTrain_SAC_0.005_0.0005_128.pt' # 97.23%
# path = 'logTrain_SAC_0.005_0.0005_256.pt' # 95.97%
path = 'logTrain_SAC_0.005_0.0005_64.pt' # 98.52%
# path = 'logTrain_SAC_0.01_0.0003_128.pt' # 97.42%
# path = 'logTrain_SAC_0.01_0.0003_256.pt' # 97.91%
# path = 'logTrain_SAC_0.01_0.0003_64.pt' # 91.26%
#path = 'logTrain_SAC_0.01_0.0005_128.pt' # 98.00%
# path = 'logTrain_SAC_0.01_0.0005_256.pt' #96.14%
# path = 'logTrain_SAC_0.01_0.0005_64.pt' # 96.30%
#env = gym.make('CartPole-v0')

state_dim = 4
action_dim = 8
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = SAC(state_size=state_dim, action_size=action_dim, device=device, tau=tau, learning_rate = learning_rate)
agent.load(path)
state = env.reset()
#print(state)
while True:
    action = agent.get_action_exploitation(state)
    #print(action)
    next_state, reward, done, info = env.step(action.item())
    state = next_state
    if done:
        break
       	
	

 