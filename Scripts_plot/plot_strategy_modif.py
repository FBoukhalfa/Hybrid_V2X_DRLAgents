with open('filePath_PPO_slides.txt') as f:
    lines = f.readlines()
    

import matplotlib.pyplot as plt

c1, c2, c3 = 0, 0, 0

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	
	if action == 1:
		color = 'red'
		c1 +=1
	if action  == 2:
		color = 'blue'
		c2 +=1
	if action == 3:
		color = 'black'
		c3 +=1
		
	if c1 == 1:
		label = 'DSRC'
		plt.scatter(state1, state2, marker = '.', color = color,label = label)
	elif c2 == 1:
		label = 'VLC HeadLight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif c3 == 1:
		label = 'DSRC and VLCHeadlight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	else:	
		plt.scatter(state1, state2, marker = '.', color = color)

print(c1)
print(c2)
print(c3)
plt.xlabel('Angle (rad)')
plt.ylabel('Distance (m)')
plt.grid()
plt.legend(loc = 2)
plt.ylim(0, 7)  # This sets the y-axis limits to go from 0 to 7
plt.show()

