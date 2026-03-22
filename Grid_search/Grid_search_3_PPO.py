import os 
import subprocess




# path = "logTrain_0.001_0.001_0.1.pt" # 97.286405159%
# path = "logTrain_0.001_0.001_0.2.pt" # 97.155126141%
# path = "logTrain_0.0001_0.001_0.1.pt" # 95.72810317%


lr_actor_list =  [1e-2, 1e-3]       
lr_critic_list = [1e-2,1e-3]
eps_clip_list =  [0.001,0.05]
  

for lr_actor in lr_actor_list:
    for lr_critic in lr_critic_list:
        for eps_clip in eps_clip_list:
                
                namefile = "logTrain_{}_{}_{}.pt".format(lr_actor, lr_critic, eps_clip)
                command = f"python3.7 ../Training_and_Test_scripts/train.py --lr_actor {lr_actor} --lr_critic {lr_critic} --eps_clip {eps_clip} --namefile {namefile}" 
                subprocess.call(command, shell=True)



