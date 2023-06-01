import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os
import neat
from gymnasium.wrappers import RecordVideo


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset(seed=10)[0].flatten()
        done = False
        fitness = 0.0
        # while not done:
        #     action = net.activate(observation)
        #     #argmax the function
        #     action = np.argmax(action)
        #     observation_, reward, done, _, info = env.step(action)
        #     observation_ = observation.flatten()
        #     fitness += reward
        #     observation = observation_
        terminated = truncated = False
        while not (terminated or truncated):
            action = net.activate(observation)
            action = np.argmax(action)
            observation_, reward, terminated, truncated, info = env.step(action)
            observation_ = observation.flatten()
            fitness += reward
            observation = observation_

        genome.fitness = fitness

def learn(env, config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 30)
    print('\nBest genome:\n{!s}'.format(winner))

    #test the best genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #show winner in action
    observation = env.reset(seed=10)[0].flatten()
    done = False
    while not done:
        action = winner_net.activate(observation)
        #argmax the function
        action = np.argmax(action)
        observation_, _, done, _, info = env.step(action)
        #normazlize the observation over 3 values
        

        observation_ = observation.flatten()
        observation = observation_
        env.render()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    config = {
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
        }
        , "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # [s]
        "lanes_count": 3,
        "initial_spacing": 2,
        "collision_reward": -5,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [20, 30],
        #reward
        "reward": {
            "type": "aggressive",
            "on_collision": -5,
            "on_lane_change": -0.1,
            "on_lane_change_success": 0.5,
            "on_right_lane": 0.1,
            "high_speed": 0.4,
            "high_accel": 0.2,
            "close_to_intersection": 0.2,
            "on_route": 0.2,
            "steering": 0.2,
            "distance_from_center": 0.2,
            "heading_difference": 0.2,
            "in_front": 0.2,
            "speed_difference": 0.2,
            "lane_difference": 0.2
            }
    }

    env = gym.make("highway-fast-v0", render_mode='rgb_array')

    # Wrap the env by a RecordVideo wrapper
    """env = RecordVideo(env, video_folder="run", episode_trigger=lambda e: True)  # record all episodes

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)"""
    env.configure(config)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ff.txt')

    learn(env, config_path)
