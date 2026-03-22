#%%
import matplotlib.pyplot as plt
import numpy as np

data = np.load('PPO_returns.npy')
plt.plot(data)


data = np.load('rainbow_rewards.npy')
plt.plot(data[0:1000])

data = np.load('SAC_rewards.npy')
plt.plot(data[0:1000])

data = np.load('TRPO_rewards.npy')
plt.plot(data[:])


#data = np.load('SAC_rewards.npy')
#plt.plot(data[0:1000])


plt.xlabel("Episode")
plt.ylabel("Cumulative rewards")

plt.legend(['PPO', 'Rainbow DQN', 'SAC', 'TRPO'])



plt.grid()
plt.savefig("Reward_banchmark.png")
plt.show()


#%%
