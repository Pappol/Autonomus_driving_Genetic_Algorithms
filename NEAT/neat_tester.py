import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import numpy as np
#import neat
import torch
import torch.nn as nn
from matplotlib import animation
import wandb
import visualize
import argparse

import neat
import warnings
warnings.filterwarnings("ignore")

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
            
        for i in range(n_training_env):
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

        genome.fitness = sum(genome_fitness) / n_training_env
        #wandb.log({"individual_fitness": genome.fitness}, commit=False)
        test_fitness.append(genome)

    #calculate mean and standard deviation of fitness
    mean_fitness = sum([x.fitness for x in test_fitness]) / len(test_fitness)
    std_fitness = np.std([x.fitness for x in test_fitness])
    wandb.log({"mean_fitness": mean_fitness, "std_fitness": std_fitness})

    test_fitness.sort(reverse=True)

    if(test_fitness[0].fitness > 30):

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
            save_frames_as_gif(frames=frames, path = save_path, filename=str(i) + '_genome_' + str(test_fitness[0].key) + "_best_agent_visualized_TR_"+str(n_training_env)+".gif")
            visualize.draw_net(config, test_fitness[0], view=False, filename= save_path + str(i) + '_genome_' + str(test_fitness[0].key) + "_winner-net_TR_"+str(n_training_env)+".gv")
            visualize.draw_net(config, test_fitness[0], view=False, filename= save_path + str(i) + '_genome_' + str(test_fitness[0].key) + "_winner-net-pruned_TR_"+str(n_training_env)+".gv", prune_unused=True)

def main(args):
    config = neat.config.Config(DriverGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                args.config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, args.generations)

    visualize.draw_net(config, winner, view=False, filename= "nets/winner-net-"+str(n_training_env)+".gv")
    visualize.draw_net(config, winner, view=False, filename= "nets/winner-net-pruned-"+str(n_training_env)+".gv", prune_unused=True)

    print('\nBest genome:\n{!s}'.format(winner))

    env_scores = []
    
    for i in range(args.test):
        observation = env.reset(seed = i)[0]
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        observation = convert_observation(observation)
        sum_reward = 0
        done = truncated = False

        while (not done) and (not truncated):
                state = torch.FloatTensor(observation)
                state = state.to(device)
                final_layer = net.activate(state)
                action = np.argmax(final_layer)
                observation_next, reward, done, truncated, info = env.step(action)
                observation = convert_observation(observation_next)
                sum_reward += reward

        print('fitness', sum_reward)
        env_scores.append([i, sum_reward])

    env_scores.sort(reverse=True)

    for i in range(0, 4):
        observation = env.reset(seed = env_scores[i][0])[0]
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
                frames.append(env.render())

        print('fitness', sum_reward)
        save_frames_as_gif(frames=frames, path = save_path, filename=str(i) + "_best_agent_visualized_"+str(n_training_env)+".gif")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NEAT algorithm for the highway environment')
    parser.add_argument('--config_path', type=str, default='config_ex.txt', 
                        help='path of the config file')
    parser.add_argument('--generations', type=int, default=100,
                        help='number of generations')
    parser.add_argument('--save_path', type=str, default='gif/',
                        help='path of the folder where to save the gif')
    parser.add_argument('--test', type=int, default=100,
                        help='number of test to perform')
    parser.add_argument('--n_training_env', type=int, default=3,
                        help='number of training environments')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    env=gym.make("highway-fast-v0", render_mode = 'rgb_array')

    env.configure({'observation': 
                {'type': 'TimeToCollision', 'horizon': 10}, 
                    'action': {'type': 'DiscreteMetaAction'}, 
                    "vehicles_count": 10,
                    'duration': 40, 
                    'lanes_count': 4, 
                    'collision_reward': -1, 
                    'high_speed_reward': 3, 
                    'reward_speed_range': [10, 30], 
                    'normalize_reward': False})

    args = parser.parse_args()
    save_path = args.save_path
    n_training_env = args.n_training_env

    #set wandb name based on number of training environments
    wandb.init(project="neat-testing", name="different-env-{}".format(n_training_env))

    main(args)
 