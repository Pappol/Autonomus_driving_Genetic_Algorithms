
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from stable_baselines3 import DQN
import gymnasium
# from HighwayEnv.highway_env.envs.highway_env import HighwayEnvCustom
from matplotlib import animation
import matplotlib.pyplot as plt


from gymnasium.envs.registration import register

register(
    id='highway-custom-v0',
    entry_point='HighwayEnv.highway_env.envs.highway_env:HighwayEnvCustom',
)


env = gymnasium.make("highway-custom-v0", render_mode="human")
# env = gymnasium.make("racetrack-v0", render_mode="human")

# config = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 2,
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-20, 20],
#             "vy": [-20, 20]
#         },
#         "absolute": False,
#         "order": "sorted",
#         "action": {
#         "type": "DiscreteMetaAction"
#     }
#     }
# }
config = {
     "vehicles_density": 1
}
env.configure(config)
obs, info = env.reset()
# print(obs)

def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


model = DQN.load("highway_dqn/model")
r = [0]
while True:
    frame = []
    tot_r = 0
    obs, info = env.reset()
    done = truncated = False
    num_steps = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        tot_r += reward
        num_steps += 1
        frame.append(env.render())
    if tot_r > 0.9*max(r):
        save_frames_as_gif(frame, "./", "gifs/%.3f.gif"%tot_r)
    r.append(tot_r)
    print(num_steps, tot_r, max(r))
