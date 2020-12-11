import gym
import numpy as np

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed

env = gym.make('gym_continuouscartpole:ContinuousCartPole-v1')

model = ReservoirSAC(MlpPolicy, env, verbose=1, buffer_size=100, batch_size=64)
model.learn(total_timesteps=1000, log_interval=4)
model.save("reservoir_sac")

del model  # remove to demonstrate saving and loading

model = SAC.load("reservoir_sac")

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
