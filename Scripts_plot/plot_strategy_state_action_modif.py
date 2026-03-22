import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax = plt.subplots()


with open('filePath_scenario_2_PPO.txt') as f:
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
	
	# generate inset axes
	plt.scatter(state1, state2, marker = '.', color = color, label = label)
	plt.xlabel('Angle (rad)')
	plt.ylabel('Distance (m)')
	plt.ylim(38,86)
	plt.xlim(-1.7,1.7)
	plt.grid()
	plt.legend(loc = 2)
	#axins = zoomed_inset_axes(ax, 1.5, loc='upper right')  # zoom = 1.5
	axins = zoomed_inset_axes(ax, 1.5, loc='upper right')  # zoom = 1.5
# plot in the inset axes
axins.scatter(state1, state2)



	
# Compute the rate of utilisation of each action (Table II of the paper)
""" print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8) """







# fix the x, y limit of the inset axes
axins.set_xlim(-1.5, -1)
axins.set_ylim(50,60)



plt.savefig("Rainbow_Action_State_Robustness.png")
plt.show()

