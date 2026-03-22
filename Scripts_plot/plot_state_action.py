import numpy as np
import matplotlib.pyplot as plt
import csv

A = [] 
X = []
Y = []

 
with open('filePath.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=' ', unpack=True)
    
    for ROWS in plotting:
	
	
        X.append(double(ROWS[2]))
        Y.append(double(ROWS[0]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
scatter = ax1.scatter(x, y, a=a)

legend1 = ax1.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax1.add_artist(legend1)


plt.show()




 
