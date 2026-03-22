import matplotlib.pyplot as plt
import numpy as np

with open("filePath_PPO_slides.txt", 'r' ) as f:
    lines = f.readlines()
    
	
c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 
label_added = [False]*8

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	
	if action == 0:
		color = 'cyan'
		c1 +=1
		label = 'No transmission' if not label_added[action] else None
		label_added[action] = True
	elif action  == 1:
		color = 'red'
		c2 +=1
		label = 'DSRC' if not label_added[action] else None
		label_added[action] = True
	elif action == 2:
		color = 'blue'
		c3 +=1	
		label = 'VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 3:
		color = 'black'
		c4 +=1
		label = 'DSRC and VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 4:
		color = 'yellow'
		c5 +=1		
		label = 'VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 5:
		color = 'green'
		c6 +=1		
		label = 'DSRC and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 6:
		color = 'silver'
		c7 +=1	
		label = 'VLCHeadlight and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 7:
		color = 'gray'
		c8 +=1
		label = 'All Access Points' if not label_added[action] else None
		label_added[action] = True
	plt.subplot(2, 2, 1)
	plt.scatter(state1, state2, s=150, marker = '.', color = color, label = label)




plt.xlabel('Angle (rad)',fontsize=22)
plt.ylabel('Distance (m)',fontsize=22)
plt.title("PPO", fontsize=22)
plt.ylim(38,86)
plt.xlim(-1.7,1.7)
plt.grid()
plt.legend(loc = 2, fontsize=16)


with open('filePath_trpo.txt') as f:
    lines = f.readlines()
    
	
c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 
label_added = [False]*8

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	
	if action == 0:
		color = 'cyan'
		c1 +=1
		label = 'No transmission' if not label_added[action] else None
		label_added[action] = True
	elif action  == 1:
		color = 'red'
		c2 +=1
		label = 'DSRC' if not label_added[action] else None
		label_added[action] = True
	elif action == 2:
		color = 'blue'
		c3 +=1	
		label = 'VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 3:
		color = 'black'
		c4 +=1
		label = 'DSRC and VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 4:
		color = 'yellow'
		c5 +=1		
		label = 'VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 5:
		color = 'green'
		c6 +=1		
		label = 'DSRC and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 6:
		color = 'silver'
		c7 +=1	
		label = 'VLCHeadlight and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 7:
		color = 'gray'
		c8 +=1
		label = 'All Access Points' if not label_added[action] else None
		label_added[action] = True
	plt.subplot(2, 2, 2)
	plt.scatter(state1, state2, s=150, marker = '.', color = color, label = label)


plt.xlabel('Angle (rad)',fontsize=22)
plt.ylabel('Distance (m)',fontsize=22)
plt.title("TRPO", fontsize=22)
plt.ylim(38,86)
plt.xlim(-1.7,1.7)
plt.grid()
plt.legend(loc = 2, fontsize=16)

with open('filePath_SAC.txt') as f:
    lines = f.readlines()
    
	
c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 
label_added = [False]*8

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	
	if action == 0:
		color = 'cyan'
		c1 +=1
		label = 'No transmission' if not label_added[action] else None
		label_added[action] = True
	elif action  == 1:
		color = 'red'
		c2 +=1
		label = 'DSRC' if not label_added[action] else None
		label_added[action] = True
	elif action == 2:
		color = 'blue'
		c3 +=1	
		label = 'VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 3:
		color = 'black'
		c4 +=1
		label = 'DSRC and VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 4:
		color = 'yellow'
		c5 +=1		
		label = 'VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 5:
		color = 'green'
		c6 +=1		
		label = 'DSRC and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 6:
		color = 'silver'
		c7 +=1	
		label = 'VLCHeadlight and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 7:
		color = 'gray'
		c8 +=1
		label = 'All Access Points' if not label_added[action] else None
		label_added[action] = True
	plt.subplot(2, 2, 3)
	plt.scatter(state1, state2, s=150, marker = '.', color = color, label = label)


plt.xlabel('Angle (rad)',fontsize=22)
plt.ylabel('Distance (m)',fontsize=22)
plt.title("SAC", fontsize=22)
plt.ylim(38,86)
plt.xlim(-1.7,1.7)
plt.grid()
plt.legend(loc = 2, fontsize=16)



with open('filePath_Rainbow.txt') as f:
    lines = f.readlines()
    
	
c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 
label_added = [False]*8

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	
	if action == 0:
		color = 'cyan'
		c1 +=1
		label = 'No transmission' if not label_added[action] else None
		label_added[action] = True
	elif action  == 1:
		color = 'red'
		c2 +=1
		label = 'DSRC' if not label_added[action] else None
		label_added[action] = True
	elif action == 2:
		color = 'blue'
		c3 +=1	
		label = 'VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 3:
		color = 'black'
		c4 +=1
		label = 'DSRC and VLCHeadlight' if not label_added[action] else None
		label_added[action] = True
	elif action == 4:
		color = 'yellow'
		c5 +=1		
		label = 'VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 5:
		color = 'green'
		c6 +=1		
		label = 'DSRC and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 6:
		color = 'silver'
		c7 +=1	
		label = 'VLCHeadlight and VLCtaillight' if not label_added[action] else None
		label_added[action] = True
	elif action == 7:
		color = 'gray'
		c8 +=1
		label = 'All Access Points' if not label_added[action] else None
		label_added[action] = True
	plt.subplot(2, 2, 4)
	plt.scatter(state1, state2, s=150, marker = '.', color = color, label = label)




plt.xlabel('Angle (rad)',fontsize=22)
plt.ylabel('Distance (m)',fontsize=22)
plt.title("Rainbow DQN", fontsize=22)
plt.ylim(38,86)
plt.xlim(-1.7,1.7)
plt.grid()
plt.legend(loc = 2, fontsize=16)
plt.savefig("Action_State_Benchmark.png")
plt.show()

