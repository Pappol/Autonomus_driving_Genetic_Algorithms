import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import numpy as np
import os
#import neat
import torch
import torch.nn as nn
from matplotlib import animation
import wandb

wandb.init(project="neat-highway")

import sys
sys.path.insert(0,'/Users/pappol/Desktop/uni/Bio inspired/project/NEAT')
import neat
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)
path = "/Users/pappol/Desktop/uni/Bio inspired/project/NEAT/"
N_GEN = 150

class DriverGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def __lt__(self, other):
        return self.fitness < other.fitness


def convert_observation(observation_matrix):
    """
    Converts observation matrix from VxLxH to L array, where the array contains the depth of the closest collision
    :param observation_matrix:
    :return:
    """
    collision_vector=[]
    for row in observation_matrix[1]:
    
            index = np.where(row > 0)[0]
            
            if len(index) > 0:
                collision_vector.append(index[0])
            else:
                collision_vector.append(-1)
    return collision_vector

def save_frames_as_gif(frames, path, filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    
    anim.save(path + filename, writer='imagemagick', fps=60)


def eval_genomes(genomes, config):
    test_fitness = []
    m = nn.Softmax(dim = 0)

    for genome_id, genome in genomes:    
        
        genome_fitness = []
        net = neat.nn.FeedForwardNetwork.create(genome, config) # net.create creates just one net
            
        for i in range(3):
            observation = env.reset()[0]
            observation = convert_observation(observation)
            is_done = terminated = False
            cumulative_fitness = 0
            while not(is_done) and not(terminated):
                state = torch.FloatTensor(observation)
                
                q_values = net.activate(state)                
                q_values_tens = torch.Tensor(q_values)
                sf_ten = m(q_values_tens)
                
                _, selected_action = torch.max(sf_ten, dim = 0)
                
                action = int(selected_action.item())  
                  
                new_state, reward, is_done, terminated, info = env.step(action) 
                
                observation = convert_observation(new_state)    
                cumulative_fitness += reward
                
            genome_fitness.append(cumulative_fitness)

        genome.fitness = sum(genome_fitness) / 3
        wandb.log({"Individual fitness": genome.fitness})
        
        test_fitness.append(genome)

    mean_fitness = sum([x.fitness for x in test_fitness]) / len(test_fitness)
    wandb.log({"Mean fitness": mean_fitness})

    test_fitness.sort(reverse=True)

    if(test_fitness[0].fitness > 17):

        for i in range(20):
            observation = env.reset(seed = i)[0]
            net = neat.nn.FeedForwardNetwork.create(test_fitness[0], config)
            observation = convert_observation(observation)
            sum_reward = 0
            done = truncated = False
            frames=[]

            while (not done) and (not truncated):
                state = torch.FloatTensor(observation)
                state = state.to(device)
                final_layer = net.activate(state)
                action = np.argmax(final_layer)
                observation_next, reward, done, truncated, info = env.step(action)
                observation = convert_observation(observation_next)
                sum_reward += reward
                env.render()
                frames.append(env.render())

            print('fitness', sum_reward)
            save_frames_as_gif(frames=frames, path = path, filename=str(i) + '_genome_' + str(test_fitness[0].key) + "_best_agent_visualized_.gif")

def run(env, config_path):
    config = neat.config.Config(DriverGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, N_GEN)

    print('\nBest genome:\n{!s}'.format(winner))
    
    for i in range(10):
        observation = env.reset(seed = i)[0]
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        observation = convert_observation(observation)
        sum_reward = 0
        done = truncated = False
        frames=[]

        while (not done) and (not truncated):
                state = torch.FloatTensor(observation)
                state = state.to(device)
                final_layer = net.activate(state)
                action = np.argmax(final_layer)
                observation_next, reward, done, truncated, info = env.step(action)
                observation = convert_observation(observation_next)
                sum_reward += reward
                env.render()
                frames.append(env.render())

        print('fitness', sum_reward)
        save_frames_as_gif(frames=frames, path = path, filename=str(i) + "_best_agent_visualized_.gif")
        

if __name__ == '__main__':
    
    env=gym.make("highway-fast-v0", render_mode = 'rgb_array')

    env.configure({'observation': 
                {'type': 'TimeToCollision', 'horizon': 10}, 
                    'action': {'type': 'DiscreteMetaAction'}, 
                    'duration': 40, 
                    'lanes_count': 4, 
                    'collision_reward': -5, 
                    'high_speed_reward': 1, 
                    'reward_speed_range': [23, 30], 
                    'normalize_reward': False})

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_ex.txt')

    run(env, config_path)
