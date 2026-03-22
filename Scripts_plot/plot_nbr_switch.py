# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# Benchmark number of switch:
# TRPO 15
# Rainbow 29
# SAC 27
# PPO 8
 
# create a DataFrame
""" personAges = pd.DataFrame({'Algorithms': ['TRPO', 'SAC',
                                      'PPO', 'Rainbow'],
                           'Number of switches': [1,2,3,4],

                           'switch': [15,26,37,47]},
                           
                           )
 
# group data & plot histogram
personAges.pivot(columns='Algorithms', values='Number of switches').plot.hist()
plt.show()
plt.savefig("Benchmark_number_switch.png") """


""" import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Algorithms = ['TRPO', 'SAC', 'PPO', 'Rainbow']
Number_of_switches = [15,27,8,29]
ax.bar(Algorithms,Number_of_switches)
plt.show()
 """


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["TRPO", "SAC", "PPO", "Rainbow DQN"])
y = np.array([15,27,8,29])
c = ['red', 'yellow', 'black', 'blue']
plt.bar(x,y,color = c)
plt.xlabel("DRL Benchmark Algorithms")
plt.ylabel("Number of switch")
plt.grid()
plt.show()
plt.savefig("Benchmark_number_switch.png")