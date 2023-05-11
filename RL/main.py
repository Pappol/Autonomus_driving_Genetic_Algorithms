import gymnasium as gym
from model import Agent
import highway_env
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from stable_baselines3 import DQN

NUMBER_EPISODES=10000 #Basically number of epochs
TIMESTEPS=100

def renderEnvironmentExample(env):
    obs, info = env.reset()
    env.render()
    env.close()

def randomChoiceDemo(env):
    for e in range(NUMBER_EPISODES):
        initial_state = env.reset()
        print(e)
        #env.render()
        appendedObservations = []
        for timeIndex in range(TIMESTEPS):
            #print(timeIndex)
            random_action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(random_action)
            appendedObservations.append(observation)
            if (terminated):
                time.sleep(0.5)
                #env.close()
                break

def list_envs():
    all_envs = gym.envs.registry

    print(sorted(all_envs))

def learnDQNetwork(env):
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=128, n_actions=5, eps_end=0.01, input_dims=25, lr=5e-4)
    scores, epshistory = [], []

    for i in range(NUMBER_EPISODES):
        score = 0
        done = False
        observation = env.reset(seed=10)[0].flatten()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            observation_ = observation.flatten()
            score += reward
            agent.store_transition(state=observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
        scores.append(score)
        epshistory.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print('episode ', i,
              '\n\tscore=', score,
              '\n\tepsilon=', agent.epsilon)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    #list_envs()
    env=gym.make("highway-fast-v0", render_mode='rgb_array')

    learnDQNetwork(env)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
