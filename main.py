import gym
import numpy as np
import torch as th

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from sac_mer import SACMER
from sac_expanded import SACExpanded
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed

env = gym.make('gym_continuouscartpole:ContinuousCartPole-v1')
# env = gym.make('Pendulum-v0')

# model_alg = ReservoirSAC
model_alg = SACMER
# model_alg = SACExpanded
optimizier_kwargs = {}
policy_kwargs = {
    'optimizer_class': th.optim.Adam,
    'optimizer_kwargs': optimizier_kwargs,
}
model = model_alg(MlpPolicy, env, verbose=2, buffer_size=10000, batch_size=64, learning_rate=3e-4, learning_starts=300,
                  gradient_steps=4, policy_kwargs=policy_kwargs, mer_s=2)
for length in [2, 0.5]:
    env.env.length = length
    model.update_env(env)
    model.learn(total_timesteps=1000, log_interval=4, reset_num_timesteps=False)
    obs = env.reset()
    count = 0
    num_of_games_played = 0
    while num_of_games_played < 10:
        count += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(f"Episode length: {count}")
            count = 0
            num_of_games_played += 1
            obs = env.reset()

model.save(model_alg.__name__)

# del model  # remove to demonstrate saving and loading
#
# model = model_alg.load(model_alg.__name__)
#
# obs = env.reset()
# count = 0
# while True:
#     count += 1
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print(f"Episode length: {count}")
#         count = 0
#         obs = env.reset()
