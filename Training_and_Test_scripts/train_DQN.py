from Agents.DQN import DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy


# hyper-parameters
BATCH_SIZE = 128
LR = 0.0001
GAMMA = 0.99
EPISILO = 0.1
DECAY = 0.99
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 500




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
ENV_A_SHAPE = 0 




dqn = DQN(env = env, BATCH_SIZE = BATCH_SIZE, LR=LR, GAMMA = GAMMA, EPISILO = EPISILO, MEMORY_CAPACITY = MEMORY_CAPACITY, Q_NETWORK_ITERATION = Q_NETWORK_ITERATION)
episodes = 2000
reward_list = []
#plt.ion()
# fig, ax = plt.subplots()
for i in range(episodes):
    state = env.reset()
    ep_reward = 0
    while True:
        action = dqn.choose_action(state, False)
        next_state, reward, done, info = env.step(action)
        

        dqn.store_transition(state, action, reward, next_state)
        ep_reward += reward

        if dqn.memory_counter >= MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
        if done:
            dqn.EPISILO = max(dqn.EPISILO*DECAY,0.05)
            break
        state = next_state
    # r = copy.copy(reward)
    reward_list.append(ep_reward)
    print(ep_reward)

       
check_point = 'model_dqn.pt'

dqn.save(check_point)       

plt.plot(reward_list)
dqn.save(args.namefile)
np.save('DQN_rewards.npy', reward_list)
#plt.xlabel("Episode")
#plt.ylabel("Reward")
#plt.grid()
#plt.show()       
 

