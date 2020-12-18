import gym
import numpy as np
import torch as th
from datetime import datetime
import subprocess

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from sac_mer import SACMER
from sac_expanded import SACExpanded
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed
from utils import change_env_parameters

NUM_OF_REDOS = 10
env_name = 'gym_continuouscartpole:ContinuousCartPole-v1'
model_algs = [ReservoirSAC, SACMER]
buffer_sizes = [10000, 1000, 100]

now = datetime.now().strftime("%Y_%m_%d__%H_%M")
save_path = './JUST_TRY_experiments__' + now + '/'




def train_evolving__eval_on_train_env(model_alg, reset_optimizers, buffer_size, subsave, iteration, last_round_no_mer):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    tensorboard_path = subsave + '/tb_' + str(iteration)

    optimizer_kwargs = {}
    policy_kwargs = {
        'optimizer_class': th.optim.Adam,
        'optimizer_kwargs': optimizer_kwargs,
    }
    model = model_alg(MlpPolicy, env, verbose=2, buffer_size=buffer_size, batch_size=64, learning_rate=3e-4,
                      learning_starts=100,
                      gradient_steps=4, policy_kwargs=policy_kwargs, mer_s=2, mer_gamma=0.3, monitor_wrapper=True,
                      tensorboard_log=tensorboard_path)

    for i_length, length in enumerate([0.5, 0.4, 0.3, 0.2]):
        parameter_dict = {'length': length}
        tb_log_name = 'run_' + str(i_length) + '_len_' + str(length)
        change_env_parameters(env, eval_env, parameter_dict=parameter_dict)
        if model_alg.__name__ == 'SACMER' and last_round_no_mer and length == 0.2:
            is_reservoir = False
            is_mer = False
        else:  # This will not have any effect on regular SAC
            is_reservoir = True
            is_mer = True
        model.update_env(env, monitor_wrapper=False, is_reservoir=is_reservoir, reset_optimizers=reset_optimizers,
                         eval_env=eval_env)  # environment already wrapped so monitor_wrapper=False
        model.learn(total_timesteps=1000, log_interval=1, reset_num_timesteps=False, eval_freq=1, n_eval_episodes=10,
                    eval_log_path=tensorboard_path + '/' + tb_log_name,
                    tb_log_name=tb_log_name, is_mer=is_mer)
        env.reset()
        eval_env.reset()
    model.save(subsave + 'model_' + str(iteration))




################################################################
# SACMER - no changes
################################################################
subsave = save_path + 'SAC_no_reset/'
model_alg = SACMER
reset_optimizers = False
buffer_size = 10000
last_round_no_mer = False
for i in range(2):
    train_evolving__eval_on_train_env(model_alg, reset_optimizers, buffer_size, subsave, i, last_round_no_mer)



