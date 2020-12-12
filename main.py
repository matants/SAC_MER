import gym
import numpy as np
import torch as th

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed

env = gym.make('gym_continuouscartpole:ContinuousCartPole-v1')
model_alg = ReservoirSAC
# model_alg = SAC
optimizier_kwargs = {}
policy_kwargs = {
    'optimizer_class': th.optim.SGD,
    'optimizer_kwargs': optimizier_kwargs,
}

model = model_alg(MlpPolicy, env, verbose=1, buffer_size=100, batch_size=64, learning_rate=3e-4, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000, log_interval=4)
model.save(model_alg.__name__)

del model  # remove to demonstrate saving and loading

model = model_alg.load(model_alg.__name__)

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
