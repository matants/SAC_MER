import gym
import numpy as np
import torch as th
from datetime import datetime
import subprocess

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from sac_mer import SACMER
from algs_expanded import SACExpanded
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed
from utils import change_env_parameters

env_name = 'gym_continuouscartpole:ContinuousCartPole-v1'
env = gym.make(env_name)
eval_env = gym.make(env_name)

# model_alg = ReservoirSAC
# model_alg = SACMER
model_alg = SACExpanded

now = datetime.now().strftime("%Y_%m_%d__%H_%M")
tensorboard_path = './tb__' + now + '/'

optimizer_kwargs = {}
policy_kwargs = {
    'optimizer_class': th.optim.Adam,
    'optimizer_kwargs': optimizer_kwargs,
}
all_result_means = []
model = model_alg(MlpPolicy, env, verbose=2, buffer_size=10000, batch_size=64, learning_rate=3e-4, learning_starts=256,
                  gradient_steps=4, policy_kwargs=policy_kwargs, mer_s=2, mer_gamma=0.3, monitor_wrapper=True,
                  tensorboard_log=tensorboard_path)
# bashCommand = "tensorboard --logdir " + tensorboard_path
# process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

for i_length, length in enumerate([0.2, 0.1]):
    parameter_dict = {'length': length}
    tb_log_name = 'run_' + str(i_length) + '_len_' + str(length)
    change_env_parameters(env, eval_env, parameter_dict=parameter_dict)
    model.update_env(env, monitor_wrapper=False, is_reservoir=True, reset_optimizers=False,
                     eval_env=eval_env)  # environment already wrapped so monitor_wrapper=False
    model.learn(total_timesteps=300, log_interval=1, reset_num_timesteps=False, eval_freq=10,
                eval_log_path=tensorboard_path + '/' + tb_log_name,
                tb_log_name=tb_log_name)
    obs = env.reset()
    count = 0
    num_of_games_played = 0
    tot_counts = []
    while num_of_games_played < 10:
        count += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            print(f"Episode length: {count}")
            tot_counts.append(count)
            count = 0
            num_of_games_played += 1
            obs = env.reset()
    print(f"Mean reward: {np.mean(tot_counts)}")
    all_result_means.append(np.mean(tot_counts))
print(f"Means of experiments: {all_result_means}")
model.save(model_alg.__name__ + '__' + now)
db = 1

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
