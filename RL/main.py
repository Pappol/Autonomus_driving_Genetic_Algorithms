import pickle
import random

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from model import Agent
import highway_env
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import wandb
from stable_baselines3 import DQN

from matplotlib import animation

NUMBER_EPISODES = 4000  # Basically number of epochs
TIMESTEPS = 100


def renderEnvironmentExample(env):
    obs, info = env.reset()
    env.render()
    env.close()

def randomChoiceDemo(env):
    observation = env.reset()[0].flatten()
    frames=[]
    done = False
    truncated = False
    actions=[0,1,2,3,4]
    while (not done) and (not truncated):
        action = random.choice(actions)
        observation_, reward, done, truncated, info = env.step(action)
        observation_ = observation.flatten()
        observation = observation_
        frames.append(env.render())
    save_frames_as_gif(frames=frames, filename="demo.gif")


def list_envs():
    all_envs = gym.envs.registry

    print(sorted(all_envs))


def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

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
            collision_vector.append(len(row))
    return collision_vector

def showDemo(agent, env):
    for i in range(10):
        frames=[]
        score=0
        observation = convert_observation(env.reset()[0])
        done = False
        truncated = False
        while (not done) and (not truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            observation_ = convert_observation(observation_)
            score += reward
            agent.store_transition(state=observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
            frames.append(env.render())
        save_frames_as_gif(frames=frames, filename=str(i)+"_best_agent_visualized_.gif")


def save_agent(best_agent):
    with open('./run/best_agent.pkl', 'wb') as outp:
        pickle.dump(best_agent, outp, pickle.HIGHEST_PROTOCOL)
    torch.save(best_agent.Q_eval.state_dict(), "best_agent_q_eval.pt")


def learnDQNetwork():
    wandb.init(project='Highway-DQN')
    config = {
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
        }
        , "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 60,  # [s]
        "lanes_count": 4,
        "collision_reward": -5,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [23, 30],
        "normalize_reward":False
    }
    env = gym.make("highway-fast-v0", render_mode='rgb_array')
    # Wrap the env by a RecordVideo wrapper
    """env = RecordVideo(env, video_folder="run",
                      episode_trigger=lambda e: True)  # record all episodes"""

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    # env.unwrapped.set_record_video_wrapper(env)
    env.configure(config)
    agent = Agent(gamma=0.9, epsilon=1.0, batch_size=128, n_actions=5, eps_end=0.01, input_dims=3, lr=5e-5,eps_dec=3e-5)
    scores, epshistory = [], []
    best_score=0
    best_agent=None

    for i in range(NUMBER_EPISODES):
        score = 0
        done = False
        truncated=False
        frames = []
        observation = convert_observation(env.reset()[0])
        while (not done) and (not truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            observation_ = convert_observation(observation_)
            score += reward
            agent.store_transition(state=observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
            #frames.append(env.render())
        scores.append(score)
        epshistory.append(agent.epsilon)
        #avg_score = np.mean(scores[-100:])
        #save_frames_as_gif(frames=frames, filename="episode" + str(i) + ".gif")

        if score>best_score:
            best_score=score
            best_agent=agent


        print('episode ', i,
              '\n\tscore=', score,
              '\n\tepsilon=', agent.epsilon)

        wandb.log({"epidose":i,"score": score, "epsilon": agent.epsilon})

    showDemo(best_agent, env)
    save_agent(best_agent)

    wandb.finish()


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print(torch.cuda.is_available())
    list_envs()

    #env = gym.make("highway-v0", render_mode='rgb_array')
    #randomChoiceDemo(env)
    learnDQNetwork()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
