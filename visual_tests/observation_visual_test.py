import multiprocessing
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

import neat
import visualize

config_env = {
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
        }
        , "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # [s]
        "lanes_count": 4,
        "collision_reward":-10,
        "high_speed_reward":1,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
        }

env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.configure(config_env)

def convert_observation(observation_matrix):
    """
    Converts observation matrix from VxLxH to L array, where the array contains the depth of the closest collision
    :param observation_matrix:
    :return:
    """
    print(observation_matrix)
    collision_vector=[]
    for row in observation_matrix[1]:
        index = np.where(row > 0)[0]
        if len(index) > 0:
            collision_vector.append(index[0])
        else: 
            collision_vector.append(-1)
    return collision_vector


#run the environment and at each step convert the observation matrix and print it
observation = env.reset()
done = truncated = False
for i in range(100):
    #input action
    action = int(input("Enter action: "))  
    observation, reward, done, truncated, info = env.step(action)
    print(convert_observation(observation))
    env.render()
    if done:
        break