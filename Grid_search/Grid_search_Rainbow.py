import os 
import subprocess




alpha_list =  [0.1, 0.3]     
beta_list = [0.5, 0.7]
prior_eps_list =  [1e-4, 1e-5] 
atom_size_list = [25, 100]





# 0.3_0.7_0.0001_100
# 0.3_0.5_0.00001_100
  

for alpha in alpha_list:
    for beta in beta_list:
        for prior_eps in prior_eps_list:
            for atom_size in atom_size_list:
                
                namefile = "logTrainRainbow_{}_{}_{}_{}.pt".format(alpha, beta, prior_eps, atom_size)
                command = f"python3.7 ../Training_and_Test_scripts/train_RainbowDQN.py  --alpha {alpha} --beta {beta} --prior_eps {prior_eps} --atom_size {atom_size} --namefile {namefile}" 
                subprocess.call(command, shell=True)



