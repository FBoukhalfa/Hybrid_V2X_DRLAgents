import numpy as np
import matplotlib.pyplot as plt

#with open("filePath_PPO_slides.txt", 'r' ) as f:
with open("Data_plot/filePath_Rainbow.txt", 'r' ) as f:
	data = f.read()

	data = data.split('\n')

	#x = [row.split(' ')[0] for row in data]
	y = [row.split(' ')[0] for row in data]
	switch = 0
	action = 0
	#print(y)
	for action in range(len(y[:-1])):
		#print(action)
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

	#leg = ax1.legend()
	
	# Make the spacing between the two axes a bit smaller
	plt.subplots_adjust(wspace=0.5)



	plt.grid()
	plt.savefig("TRPO_Action_Time.png")
	# plt.savefig("SAC_Action_Time.png")
	# plt.savefig("Rainbow_Action_Time.png")
	# plt.savefig("PPO_Action_Time.png")
	plt.show()
