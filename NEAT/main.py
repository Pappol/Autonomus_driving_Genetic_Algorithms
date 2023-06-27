import gymnasium as gym
import torch
import numpy as np
import os
import neat
import warnings
warnings.filterwarnings("ignore")
from neat import parallel

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

def eval_genomes(genomes, config):
    #observation = env.reset(seed=1)[0].flatten()
    #total_rewards = []
        
        for genome_id, genome in genomes:
            #print('GENOME', genome_id)
            #print(genome)
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
                
                #observation = observation.flatten()
            #env.close()
            #print('gain', gain)
            genome.fitness = fitness
        
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
    print('\nBest genome:\n{!s}'.format(winner))

    #test the best genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #show winner in action
    observation = env.reset(seed=42)[0].flatten()
    done = False
    while not(is_done) and not(terminated):
            
        state = torch.FloatTensor(observation)
        q_values = winner_net.activate(state)                
        q_values_tens = torch.Tensor(q_values)

        _, selected_action = torch.max(q_values_tens, dim = 0)
        action = int(selected_action.item())
        
        new_state, reward, is_done, terminated, info = env.step(action)
        # if the current episode has ended
        observation = convert_observation(new_state)    
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
        "lanes_count": 4,
        "collision_reward":-10,
        "high_speed_reward":1,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
        }

    env = gym.make("highway-fast-v0", render_mode='rgb_array')
    env.configure(config)
    env.configure({'observation': 
                {'type': 'TimeToCollision', 'horizon': 10}, 
                    'action': {'type': 'DiscreteMetaAction'}, 
                    'duration': 40, 'lanes_count': 4, 'collision_reward': -10, 'high_speed_reward': 1, 'reward_speed_range': [23, 30], 'normalize_reward': False})


    # Wrap the env by a RecordVideo wrapper
    """env = RecordVideo(env, video_folder="run", episode_trigger=lambda e: True)  # record all episodes

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)"""
    

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ff.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    evaluator = parallel.ParallelEvaluator(24, eval_genome)
    winner = p.run(evaluator.evaluate, 2)
    print('\nBest genome:\n{!s}'.format(winner))
    #show best genome in action
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    observation = env.reset(seed=10)[0]
    observation = convert_observation(observation)

    is_done = terminated = False
            
    while not(is_done) and not(terminated):

        state = torch.FloatTensor(observation)
                    
        q_values = winner_net.activate(state)                
        q_values_tens = torch.Tensor(q_values)
        
        _, selected_action = torch.max(q_values_tens, dim = 0)
        
        action = int(selected_action.item())
      
        new_state, reward, is_done, terminated, info = env.step(action)
        observation = convert_observation(new_state)
        
        env.render()