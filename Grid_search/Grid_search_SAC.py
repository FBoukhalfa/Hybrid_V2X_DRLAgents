import os 
import subprocess

tau_list = [1e-2, 0.005]
learning_rate_list =  [5e-4, 0.0003]
batch_size_list =  [256, 128, 64]
  

for tau in tau_list:
    for learning_rate in learning_rate_list:
        for batch_size in batch_size_list:
                
                namefile = "logTrain_SAC_{}_{}_{}.pt".format(tau, learning_rate, batch_size)
                command = f"python3.7 ../Training_and_Test_scripts/train_SAC.py --tau {tau} --learning_rate {learning_rate} --batch_size {batch_size} --namefile {namefile}" 
                subprocess.call(command, shell=True)



