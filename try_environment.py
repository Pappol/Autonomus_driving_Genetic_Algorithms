import gymnasium as gym

env = gym.make("highway-v0")
env.configure({
    "manual_control": True,
    "offroad_terminal": True,
    "normalize_reward":True
})
env.reset()
done = False
sum=0
while not done:
    observation_next, reward, done,truncated, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    env.render()
    sum+=reward
    print(sum)