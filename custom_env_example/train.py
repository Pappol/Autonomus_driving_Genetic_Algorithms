import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
from stable_baselines3.common.save_util import load_from_zip_file

class CustomCallback(BaseCallback):
    def __init__(
        self,
        num_rolls=20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.num_rolls=num_rolls
        self.T = time.time()
        self._reset_count()

    def _reset_count(self):
        self.count = 0

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        self.count+=1
        if self.count % self.num_rolls == 0:
          print("saved: %.2f"%(time.time()-self.T), self.count)
          model.save("highway_dqn/model_tmp")
          self._reset_count()
          pass 

from gymnasium.envs.registration import register
import glob
def sort_by(name):
    return int(name.split("_")[-1].split(".")[0])

register(
    id='highway-custom-v0',
    entry_point='HighwayEnv.highway_env.envs.highway_env:HighwayEnvCustom',
)

env = gym.make("highway-custom-v0")
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              tau=1,
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=500,
              batch_size=64,
              gamma=0.9,
              train_freq=(5, "step"),
              gradient_steps=2,
              exploration_initial_eps=0.15,
              exploration_fraction=0.5,
              exploration_final_eps=0.05,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
path = sorted(glob.glob("highway_dqn/DQN_*"), key=sort_by)[-1]
print("Previous path %s"%path)

# load model params
data, params, pytorch_variables = load_from_zip_file(
    "highway_dqn/DQN_2/model",
    device="auto",
    custom_objects=None,
    print_system_info=False,
)
model.set_parameters(params, exact_match=True, device="auto")
model.learn(200000, log_interval=10, callback=CustomCallback(100))

path = sorted(glob.glob("highway_dqn/DQN_*"), key=sort_by)[-1]

model.save(path+"/model")
print("Model saved to %s"%path)
