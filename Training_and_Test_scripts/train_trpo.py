from Agents.TRPO.TRPO_new import *
import tqdm
import argparse

parser = argparse.ArgumentParser()
    
parser.add_argument('--delta', type=float)
parser.add_argument('--depth', type=int)
parser.add_argument('--line_search_max_iter', type=int)
parser.add_argument('--namefile', type=str)
args = parser.parse_args()

gym.register(
    id="veins-v1",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../scenario",

        "run_veins": True,  # do not start veins through Veins-Gym
    "print_veins_stdout": False,  # enable (debug) output of veins
    "port": 5561,  # pick a port to use
    "timeout": 10.0,  # new timeout value (in seconds)
    },
)

env = gym.make("veins-v1")


#env = gym.make("CartPole-v0") ### Instance of Veins gym 
#NUM_ACTIONS = env.action_space.n
NUM_ACTIONS = 8
#NUM_STATES = env.observation_space.shape[0]
NUM_STATES = 4



#env = gym.make("CartPole-v0").unwrapped


#---------------- HYPERPARAMETERS TO CHANGE --------------------------------------------
learning_rate = 1e-3 
#delta = 0.005 # 0.01
#depth = 32  # 64
#line_search_max_iter = 10 # 20
#--------------------------------------------------------------------------------------
agent_trpo = TRPO(env, NUM_STATES, NUM_ACTIONS, gamma=0.99, learning_rate=learning_rate, delta= args.delta, depth = args.depth , line_search_max_iter = args.line_search_max_iter)
ep_rewards = []
total_episode = 2000

for i in tqdm.tqdm(range(1, total_episode+1)):
    state = env.reset()
    rewards = []

    while True:
        action = agent_trpo.get_action(state, False)
        next_state, reward , done, _ = env.step(action)

        agent_trpo.memory.add(state, action, reward, next_state, done)             
        rewards.append(reward)
            
        if done:
            agent_trpo.learn()
            ep_rewards.append(sum(rewards))
                
            if i % 20 == 0:
                print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
            break

        state = next_state
    
	
#check_point = 'model_TRPO.pt'
#path = args.namefile
#agent_trpo.save(path)   
path = args.namefile
np.save(path, ep_rewards) 
#np.save('TRPO_rewards.npy', ep_rewards) 
#agent_trpo.save(check_point)   