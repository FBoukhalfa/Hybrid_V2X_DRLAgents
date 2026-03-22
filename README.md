Test VHO decision-making procedure 
================================================
This repository contains code to train, evaluate different Deep Reinforcement Learning (DRL) algorithms (PPO, SAC, TRPO, DQN, etc.) in vehicular communication domain. The focus is to maximize the transmission reliability and resource efficiency, which are critical parameters for the handover decision. We provide also some facilities such as script to save and parse data (Packet Delivery Ratio (PDR), Action versus time, Agent strategy, etc.), pretrained model, fine-tunning, robustenss test. To be able to use this code, you should first clone this respository [Vehicular Simulator](https://github.com/FouziBoukhalfa/DRL-V2X-Vertical-Handover-Benchmark/tree/master) which contains the simulator environement and then connect them through VEINS-GYM.

The following sections describ how the code is structured:

## Training & Evaluation

*The pretrained model respectively for each algorithm can be found under ```Models/``` folder.<br>
*To launch the trainning there are two methods possible : the manual training is under ```Training_and_Test_scripts/```, or by lanching the best model for each algorithm of the benchmark ```Benchamrk/```. 
## Hyperparameters Tuning
To fine tune our algorithm, we choose to adopt a grid search method, the code for this is available under ```Grid_search/``` folder.

## Data visualization
All data files got from the simulator are stored in ```Script_plot/Data_plot/```  , the script to plot the different results are under ```Script_plot/``` 
## Future work

## Contact
Fouzi Boukhalfa, Reda Alami, Mastane Achab.
fouzi.boukhalfa@tii.ae, reda.alami@tii.ae, mastane.achab@tii.ae.


## Reference
Fouzi Boukhalfa, Réda Alami, Mastane Achab, Eric Moulines, Mehdi Bennis:
Deep Reinforcement Learning Algorithms for Hybrid V2X Communication: A Benchmarking Study. CoRR abs/2310.03767 (2023)
