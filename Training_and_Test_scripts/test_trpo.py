from scenario.Delivrable_Github.Agent.TRPO.TRPO_new import *


gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

    "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": True,  # enable (debug) output of veins
    "port": 5558,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")


#env = gym.make("CartPole-v0") ### Instance of Veins gym 
#NUM_ACTIONS = env.action_space.n
NUM_ACTIONS = 8
#NUM_STATES = env.observation_space.shape[0]
NUM_STATES = 4


#check_point = 'model_TRPO.pt'
# check_point = 'logTrain_TRPO_0.005_32_10.pt' # 91.43%
# check_point = 'logTrain_TRPO_0.005_32_20.pt' # 95.15%
#check_point = 'logTrain_TRPO_0.005_64_10.pt' # 95.27%
check_point = 'logTrain_TRPO_0.005_64_20.pt' # 95.51% 
# check_point = 'logTrain_TRPO_0.01_32_10.pt' # 91.86%
# check_point = 'logTrain_TRPO_0.01_32_20.pt' # 91.97%
# check_point = 'logTrain_TRPO_0.01_64_10.pt' # 94.45% 
# check_point = 'logTrain_TRPO_0.01_64_20.pt' # 92.66%
e = check_point.split('_')
delta = float(e[2])
depth = int(e[3])
line_search_max_iter = float(e[4].split('.')[0])
#env = gym.make("CartPole-v0").unwrapped
learning_rate = 1e-3 
# agent_trpo = TRPO(env, NUM_STATES, NUM_ACTIONS, gamma=0.99, learning_rate=1e-8, delta=0.005)
# agent_trpo = TRPO(env, NUM_STATES, NUM_ACTIONS, gamma=0.99, learning_rate=1e-2, delta=0.005) # learning_rate=1e-8, 1e-4
agent_trpo = TRPO(env, NUM_STATES, NUM_ACTIONS, gamma=0.99, learning_rate=learning_rate, delta= delta, depth = depth , line_search_max_iter = line_search_max_iter)

agent_trpo.load(check_point)

ep_rewards = deque(maxlen=20)



state = env.reset()
rewards = []

while True:
    action = agent_trpo.get_action(state, True)
    next_state, reward , done, _ = env.step(action)

    agent_trpo.memory.add(state, action, reward, next_state, done)             
    rewards.append(reward)
            
    if done:
       break

    state = next_state
    
	


