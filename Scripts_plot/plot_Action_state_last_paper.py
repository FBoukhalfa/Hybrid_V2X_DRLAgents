import numpy as np
import matplotlib.pyplot as plt

with open("filePath_PPO_slides.txt", 'r' ) as f:
	data = f.read()
	data = data.split('\n')

	y = [int(row.split(' ')[0]) for row in data[0:-1]]

	switch = 0
	for action in range(len(y[:-1])):
		if y[action] != y[action + 1]:
			switch +=1
		
	print(switch)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	#ax1.set_title("Learned strategy")    
	ax1.set_xlabel('Time',fontweight='bold')
	ax1.set_ylabel('Action',fontweight='bold')

	t = range(len(y))

	# Getting unique y values and mapping y values to new set of evenly spaced values
	unique_y = np.unique(y)
	mapping = {val: i for i, val in enumerate(unique_y)}
	y_mapped = [mapping[val] for val in y]

	ax1.plot(t[:-1], y_mapped[:-1], c='r', linewidth=1)  # Adjust linewidth here

	# Setting y-axis limits explicitly
	ax1.set_ylim([0, max(y_mapped) + 1]) # +1 for visibility

	# Setting y-tick labels to show the original action values
	ax1.set_yticks(range(len(unique_y)))
	ax1.set_yticklabels(unique_y)
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	plt.ylim(0,2) 

	plt.subplots_adjust(wspace=0.5)

	plt.grid()
	plt.savefig("PPO_Action_Time_Last.png")
	plt.show()
