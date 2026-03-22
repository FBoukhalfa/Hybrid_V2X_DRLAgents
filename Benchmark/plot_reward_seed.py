#%%
""" This script is to plot the convergence curves of the different DRL of the benchmark """  


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

data_0 = np.load('Data_plot/PPO_returns_seed_1.npy')
#plt.plot(data_0)

data_1 = np.load('Data_plot/PPO_returns_seed_2.npy')
#plt.plot(data_1)

data_2 = np.load('Data_plot/PPO_returns_seed_3.npy')
#plt.plot(data_2)

data_3 = np.load('Data_plot/PPO_returns_seed_4.npy')
#plt.plot(data_3)

data_4 = np.load('Data_plot/PPO_returns_seed_5.npy')
#plt.plot(data_4)

M = np.vstack((data_0, data_1, data_2, data_3, data_4))
mean_value = np.mean(M, axis=0)
std_value = np.std(M,axis=0)
interval = 1.96 * std_value/ np.sqrt(5)

plt.plot(mean_value, color='b', label = 'PPO')
plt.fill_between(range(mean_value.shape[0]), mean_value - interval, mean_value + interval, color='b', alpha=0.2)
                                      


data_0 = np.load('Data_plot/SAC_returns_seed_1.npy')
#plt.plot(data)

data_1 = np.load('Data_plot/SAC_returns_seed_2.npy')
#plt.plot(data)

data_2 = np.load('Data_plot/SAC_returns_seed_3.npy')
#plt.plot(data)

M = np.vstack((data_0, data_1, data_2))
mean_value = np.mean(M, axis=0)
std_value = np.std(M,axis=0)
interval = 1.96 * std_value/ np.sqrt(3)

plt.plot(mean_value, color='g', label = 'SAC')
plt.fill_between(range(mean_value.shape[0]), mean_value - interval, mean_value + interval, color='g', alpha=0.2)

data_0 = np.load('Data_plot/Rainbow_returns_seed_2.npy')
# plt.plot(data)


data_1 = np.load('Data_plot/Rainbow_returns_seed_3.npy')
# plt.plot(data)

data_1 = np.load('Data_plot/Rainbow_returns_seed_4.npy')
# plt.plot(data)

M = np.vstack((data_0, data_1))
mean_value = np.mean(M, axis=0)
std_value = np.std(M,axis=0)
interval = 1.96 * std_value/ np.sqrt(3)

plt.plot(mean_value, color='r', label = 'Rainbow')
plt.fill_between(range(mean_value.shape[0]), mean_value - interval, mean_value + interval, color='r', alpha=0.2)


data_1 = np.load('Data_plot/TRPO_returns_seed_1.npy')
#plt.plot(data)

data_2 = np.load('Data_plot/TRPO_returns_seed_2.npy')
#plt.plot(data)

data_3 = np.load('Data_plot/TRPO_returns_seed_4.npy')

data_4 = np.load('Data_plot/TRPO_returns_seed_5.npy')

M = np.vstack((data_1, data_2, data_3, data_4))
mean_value = np.mean(M, axis=0)
std_value = np.std(M,axis=0)
interval = 1.96 * std_value/ np.sqrt(4)

plt.plot(mean_value, color='y', label = 'TRPO' )
plt.fill_between(range(mean_value.shape[0]), mean_value - interval, mean_value + interval, color='y', alpha=0.2)

plt.xlabel("Episode")
plt.ylabel("Cumulative rewards")

plt.legend()
# ['PPO','SAC','Rainbow','TRPO']



plt.grid()
plt.savefig("Image_output/PPO_SAC_Rainbow_TRPO_seed2.png")
plt.show()


#%%
