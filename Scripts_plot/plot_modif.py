import numpy as np
import matplotlib.pyplot as plt

with open("Data_plot/filePath_Ranbow.txt", 'r' ) as f:
	data = f.read()
	data = data.split('\n')

	y = [int(row.split(' ')[0]) for row in data]

	switch = 0
	for action in range(len(y[:-1])):
		if y[action] != y[action + 1]:
			switch +=1

	print(switch)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.set_title("Learned strategy")    
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Action')

	t = range(len(y))
	ax1.plot(t[:-1], y[:-1], c='r')

	# Setting y-axis limits explicitly
	#ax1.set_ylim([min(y), max(y)]) 

	plt.subplots_adjust(wspace=0.5)

	plt.grid()
	#plt.savefig("TRPO_Action_Time.png")
	plt.show()

