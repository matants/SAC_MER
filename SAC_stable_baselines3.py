import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make('gym_continuouscartpole:ContinuousCartPole-v1')

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=4)
model.save("sac")

del model  # remove to demonstrate saving and loading

model = SAC.load("sac")

obs = env.reset()
count = 0
while True:
    count += 1
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print(f"Episode length: {count}")
        count = 0
        obs = env.reset()
