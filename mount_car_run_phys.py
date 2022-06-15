import sys
sys.path.insert(1, 'D:/Users/Josh/github/individual_project/simulation')
from phys_utils import *
import random
import time
import numpy as np
import gym
import h5py

env = gym.make('MountainCar-v0')
skip_actions = 20
discrete_os_size = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size

action_to_key_arr = ['UP', 'DOWN', 'RIGHT']
key_to_action_dict = {'UP': [0], 'DOWN': [1], 'RIGHT': [2]}
current_letter = 'DOWN'
goal_letter = 'DOWN'
env_r = phys_discrete_arrow_env_pong(current_letter, goal_letter)

agent_dir = "D:/Users/Josh/github/individual_project/physical/phys_agents/arrow_phys_Dueling Double Per.h5"
agent = Dueling_Per_DDQNAgent(env_r, rl_params)
agent.load_model(agent_dir)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

hf = h5py.File('D:\Josh\github\individual_project\simulation\sim_agents\mount_cart.h5', 'r')
n1 = hf.get('q_table')
q_table = np.array(n1)

for episode in range(1, 2):
    discrete_state = get_discrete_state(env.reset())
    done = False
    skip_counter = 0

    print('Episode is ', episode)

    while not done:
        # Mountain car Agent acting #
        if skip_counter % skip_actions == 0:
            skip_counter = 0
            action = np.argmax(q_table[discrete_state]) # mountain car q-table predicts action
            goal_letter = action_to_key_arr[action_p[0]] # action to goal key for robot
            current_state_r = env_r.reset(current_letter, goal_letter) # reset robot env for goal key
            action_r = agent.act(current_state_r) # predict action for robot using new goal key
            new_state_r, reward, done, _ = env_r.step(action_r, steps=0) # execute action/ move to key

            current_letter = coords_to_letter(env_r.current_coords) # find current key
            if current_letter in key_to_action_dict: # if current key has an associated action..
                action_m = key_to_action_dict[current_letter] # ...use said action
            else:
                action_m = [random.randint(0, len(key_to_action_dict)-1)] # else chose a random action

        new_state, reward, done, info = env.step(action_m) # update mountain car env
        new_discrete_state = get_discrete_state(new_state) # turn new state into a discrete space

        env.render()

        discrete_state = new_discrete_state
        skip_counter += 1
