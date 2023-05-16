import gymnasium as gym
from model import Agent
import highway_env
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os
import neat


def renderEnvironmentExample(env):
    obs, info = env.reset()
    env.render()
    env.close()


def list_envs():
    all_envs = gym.envs.registry
    print(sorted(all_envs))

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset(seed=10)[0].flatten()
        done = False
        fitness = 0.0
        while not done:
            action = net.activate(observation)
            #argmax the function
            action = np.argmax(action)
            observation_, reward, done, _, info = env.step(action)
            observation_ = observation.flatten()
            fitness += reward
            observation = observation_
        genome.fitness = fitness

def learnDQNetwork(env, config_path):
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=128, n_actions=5, eps_end=0.01, input_dims=25, lr=5e-4)
    scores, epshistory = [], []

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    #list_envs()
    env=gym.make("highway-fast-v0", render_mode='rgb_array')
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ff.txt')

    learnDQNetwork(env, config_path)
