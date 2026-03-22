import os 
import subprocess

delta_list = [0.005,0.01]
#delta_list = [0.01]
depth_list = [32,64]
line_search_max_iter_list = [10,20]


for delta in delta_list:
    for depth in depth_list:
        for line_search_max_iter in line_search_max_iter_list:
                
                namefile = "logTrain_TRPO_{}_{}_{}.pt".format(delta, depth, line_search_max_iter)
                command = f"python3.7 ../Training_and_Test_scripts/train_trpo_new.py --delta {delta} --depth {depth} --line_search_max_iter {line_search_max_iter} --namefile {namefile}" 
                subprocess.call(command, shell=True)


