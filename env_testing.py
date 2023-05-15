import gymnasium as gym
from matplotlib import pyplot as plt

env = gym.make('highway-v0', render_mode='rgb_array')
env.reset()
for _ in range(100):
    action = env.action_type.actions_indexes["SLOWER"]

    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()
