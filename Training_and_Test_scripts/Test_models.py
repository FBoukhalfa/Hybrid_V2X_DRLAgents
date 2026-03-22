import os 
import subprocess


lr_actor_list =  [1e-5,1e-4,1e-3]       # learning rate for actor network ==> 0.0003 ==> pdr = 94.000%
lr_critic_list = [1e-5,1e-4,1e-3]
eps_clip_list =  [0.1,0.2,0.3]
  

for lr_actor in lr_actor_list:
    for lr_critic in lr_critic_list:
        for eps_clip in eps_clip_list:
                
                namefile = "logTrain_{}_{}_{}.pt".format(lr_actor, lr_critic, eps_clip)
                command = f"python3.7 test.py --namefile {namefile}" 
                subprocess.call(command, shell=True)



