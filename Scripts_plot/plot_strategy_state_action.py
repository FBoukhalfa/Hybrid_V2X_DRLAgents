import matplotlib.pyplot as plt

with open('filePath_SAC.txt') as f:
    lines = f.readlines()
    
	
c1, c2, c3, c4, c5, c6, c7, c8 = 0, 0, 0, 0, 0, 0, 0, 0 

for line in lines:
	e = line.split(" ")	
	action = int(e[0])
	state1 = float(e[1])
	state2 = float(e[2])
	


	if action == 0:
		color = 'red'
		c1 +=1
	if action  == 1:
		color = 'blue'
		c2 +=1
	if action == 2:
		color = 'black'
		c3 +=1	
	if action == 3:
		color = 'yellow'
		c4 +=1
	if action == 4:
		color = 'cyan'
		c5 +=1		
	if action == 5:
		color = 'green'
		c6 +=1		
	if action == 6:
		color = 'silver'
		c7 +=1	
	if action == 7:
		color = 'gray'
		c8 +=1
	
		
	if action == 0:
		label = 'No transmission'
		plt.scatter(state1, state2, marker = '.', color = color,label = label)
	elif action == 1:
		label = 'DSRC'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 2:
		label = 'VLCHeadlight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 3:
		label = 'DSRC and VLCHeadlight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 4:
		label = 'VLCtaillight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 5:
		label = 'DSRC and VLCtaillight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 6:
		label = 'VLCHeadlight and VLCtaillight'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)
	elif action == 7:
		label = 'All Access Points'
		plt.scatter(state1, state2, marker = '.', color = color, label = label)


# Compute the rate of utilisation of each action (Table II of the paper)
print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8)

plt.xlabel('Angle (rad)')
plt.ylabel('Distance (m)')
plt.grid()
plt.legend(loc = 2)
#plt.savefig("SAC_Action_State.png")
plt.show()
