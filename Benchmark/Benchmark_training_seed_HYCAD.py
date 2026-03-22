import os 
import subprocess


""" This script is to train the best model of the benchmark, use plot_reward_seed.py to see
the convergence curves"""

v_list = [1,2,3,4,5] 
#PPO
for v in v_list:
            # Best model
            # path = "logTrain_0.01_0.001_0.01.pt" # 98.495836691%
            # path = "logTrain_0.01_0.01_0.02.pt" # 97.57%s
            namefile = "PPO_returns_seed_{}.npy".format(v)
            command = f"python3.8 ../Training_and_Test_scripts/train_PPO.py --lr_actor {0.01} --lr_critic {0.01} --eps_clip {0.02} --namefile {namefile}" 
            subprocess.call(command, shell=True)



e_list = [1,2,3,4,5] 
#SAC
for e in e_list:
            # Best model
            # path = 'logTrain_SAC_0.005_0.0005_64.pt'# 98.52%
            namefile = "SAC_returns_seed_{}.npy".format(e)
            command = f"python3.8 ../Training_and_Test_scripts/train_SAC.py --tau {0.005} --learning_rate {0.0005} --batch_size {64} --namefile {namefile}" 
            subprocess.call(command, shell=True) 


i_list = [1,2,3,4,5] 

#Rainbow
for i in i_list:
                
                # Best model
                # path = "logTrainRainbow_0.3_0.7_0.0001_25.pt" # 97.57%
                namefile = "Rainbow_returns_seed_{}.npy".format(i)
                command = f"python3.8 ../Training_and_Test_scripts/train_RainbowDQN.py  --alpha {0.3} --beta {0.7} --prior_eps {0.0001} --atom_size {25} --namefile {namefile}" 
                subprocess.call(command, shell=True) 



d_list = [1,2,3,4,5] 

#TRPO
for d in d_list:
                # Best model
                #check_point = 'logTrain_TRPO_0.005_64_20.pt' # 95.51% 
                namefile = "TRPO_returns_seed_{}.npy".format(d)
                command = f"python3.8 ../Training_and_Test_scripts/train_trpo.py --delta {0.005} --depth {64} --line_search_max_iter {20} --namefile {namefile}" 
                subprocess.call(command, shell=True)
