import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from model import Agent
import highway_env
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from stable_baselines3 import DQN

from matplotlib import animation

NUMBER_EPISODES = 10000  # Basically number of epochs
TIMESTEPS = 100


def renderEnvironmentExample(env):
    obs, info = env.reset()
    env.render()
    env.close()

def randomChoiceDemo(env):
    for e in range(NUMBER_EPISODES):
        initial_state = env.reset()
        print(e)
        # env.render()
        appendedObservations = []
        for timeIndex in range(TIMESTEPS):
            # print(timeIndex)
            random_action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(random_action)
            appendedObservations.append(observation)
            if (terminated):
                time.sleep(0.5)
                # env.close()
                break


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


def learnDQNetwork():
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
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [20, 30]
    }
    env = gym.make("highway-v0", render_mode='rgb_array')
    # Wrap the env by a RecordVideo wrapper
    """env = RecordVideo(env, video_folder="run",
                      episode_trigger=lambda e: True)  # record all episodes"""

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    # env.unwrapped.set_record_video_wrapper(env)
    env.configure(config)
    agent = Agent(gamma=0.9, epsilon=1.0, batch_size=128, n_actions=5, eps_end=0.01, input_dims=90, lr=5e-4)
    scores, epshistory = [], []

    for i in range(NUMBER_EPISODES):
        score = 0
        done = False
        frames = []
        observation = env.reset()[0].flatten()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            observation_ = observation.flatten()
            score += reward
            agent.store_transition(state=observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
            frames.append(env.render())
        scores.append(score)
        epshistory.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        save_frames_as_gif(frames=frames, filename="episode" + str(i) + ".gif")

        print('episode ', i,
              '\n\tscore=', score,
              '\n\tepsilon=', agent.epsilon)


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print(torch.cuda.is_available())
    # list_envs()

    learnDQNetwork()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
