import gymnasium as gym
import torch
import numpy as np
import os
import wandb
import neat
import gym.wrappers

import warnings
warnings.filterwarnings("ignore")
from neat import parallel
import matplotlib.pyplot as plt
from matplotlib import animation

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
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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


def evaluateModel(model, env):
    """
    Evaluate the model on 10 different scenarios, saves gifs and stores score results in a txt file
    :param model:
    :param env:
    :return:
    """
    results=[]
    for i in range(10):
        observation = env.reset()[0]
        observation = convert_observation(observation)
        sum_reward = 0
        done = False
        truncated = False
        frames=[]
        while (not done) and (not truncated):
            state = torch.FloatTensor(observation)
            state = state.to(device)
            final_layer = model(state)
            output = final_layer.cpu().detach().numpy()
            action = np.argmax(output)
            observation_next, reward, done, truncated, info = env.step(action)
            observation = convert_observation(observation_next)
            sum_reward += reward
            frames.append(env.render())
        save_frames_as_gif(frames=frames, filename=str(i) + "_best_agent_visualized_.gif")
        results.append(sum_reward)
    with open('score_results.txt', 'w') as f:
        for result_idx in range(len(results)):
            string="Scenario "+str(result_idx)+" score: "+str(results[result_idx])+"\n"
            f.write(string)


def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    agents and sets their fitness
    """
    #env definition
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

    nets = []
    ge = []
    win=0
    for genome_id, genome in genomes:
        genome.fitness = 0.0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        env.reset(seed=10)
        observation = env.reset(seed=10)[0]
        observation = convert_observation(observation)

        is_done = terminated = False
        while not(is_done) and not(terminated):
            state = torch.FloatTensor(observation)
            q_values = net.activate(state)
            q_values_tens = torch.Tensor(q_values)
            _, selected_action = torch.max(q_values_tens, dim = 0)

            action = int(selected_action.item())
            new_state, reward, is_done, terminated, info = env.step(action)

            # if the current episode has ended
            observation = convert_observation(new_state)

            genome.fitness += reward

        if genome.fitness > win:
            win = genome.fitness
        
    WIN = win
    return win


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config) # net.create creates just one net
    fitness = 0.0
    observation = env.reset(seed=10)[0]
    observation = convert_observation(observation)
            
    is_done = terminated = False
            
    while not(is_done) and not(terminated):
        #print(observation)
        state = torch.FloatTensor(observation)
                
        #print(observation_idx)
                
        q_values = net.activate(state)                
        q_values_tens = torch.Tensor(q_values)

        _, selected_action = torch.max(q_values_tens, dim = 0)
                
        #print(_, selected_action)
        action = int(selected_action.item())
                
                    
        new_state, reward, is_done, terminated, info = env.step(action)
                
        # if the current episode has ended
        observation = convert_observation(new_state)    
                    
        fitness += reward        

    return fitness

def learn(env, config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 50)
    evaluateModel(winner, env)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ff.txt')

    learn(env, config_path)