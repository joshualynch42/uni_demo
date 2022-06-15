import sys
import random
import time
import numpy as np
import gym
import keyboard


env = gym.make('MountainCar-v0')

env.reset()
done = False
step = 0
env._max_episode_steps = 1000
while not done:

    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        action = 0
    elif keyboard.is_pressed('e'):  # if key 'q' is pressed
        action = 2
    else:
        action = 1

    new_state, reward, done, info = env.step(action) # update mountain car env
    position, velocity = new_state
    # print(bool(position >= env.goal_position and velocity >= env.goal_velocity))
    step += 1
    print(action)
    env.render()
