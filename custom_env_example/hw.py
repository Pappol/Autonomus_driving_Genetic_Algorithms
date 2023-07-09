
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from stable_baselines3 import DQN
import gymnasium
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium.envs.registration import register

from agent_test import dt
register(
    id='highway-custom-v0',
    entry_point='HighwayEnv.highway_env.envs.highway_env:HighwayEnvCustom',
)

env = gymnasium.make("highway-v0",render_mode="rgb_array")
env.configure(
        {
        'lanes_count': 3,
        'duration': 40, 
        'collision_reward': -5, 
        'high_speed_reward': 1, 
        'reward_speed_range': [23, 30], 
        'normalize_reward': False})
model = DQN('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]))
data, params, pytorch_variables = load_from_zip_file(
     "model_tmp.zip",
     device="auto",
     custom_objects=None,
     print_system_info=False,
)
model.set_parameters(params, exact_match=True, device="auto")

ACTIONS_ALL = {
    "Left":  0,
    "Idle":  1,
    "Right": 2,
    "Faster":3,
    "Slower":4,
}

tmp = []

for _ in range(100):
    tot_r = 0
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        # action, _states = model.predict(obs, deterministic=True)
        action = ACTIONS_ALL[dt(obs)]
        obs, reward, done, truncated, info = env.step(action)
        tot_r += reward
        env.render()
    print(tot_r)
    tmp.append(tot_r)
breakpoint()
