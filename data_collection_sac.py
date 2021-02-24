import random
import gym
import numpy as np
import torch as th
from datetime import datetime
import subprocess
from time import time

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from sac_mer import SACMER
from algs_expanded import SACExpanded
from stable_baselines3.sac import MlpPolicy
from utils import change_env_parameters, Param, AlternatingParamsUniform, SequentialParams, seed_all
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EventCallback
import pickle
import os

initial_seed = 73
NUM_OF_REDOS = 10  # how many times we run the training loops (for confidence bounds)
EVAL_FREQ = 100
N_EVAL_EPISODES = 5
NUM_TRAINING_ENVS = 4
MER_S = 2
MER_GAMMA = 0.3
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
LEARNING_STARTS = 256
GRADIENT_STEPS = 5
META_TRAINING_TIMESTEPS = 10000
FINAL_TRAINING_TIMESTEPS = 10000

env_name = 'ContinuousCartPole-v1'
buffer_sizes = [50000]

now = datetime.now().strftime("%Y_%m_%d__%H_%M")
save_path = './experiments__' + now + '/'
os.mkdir(save_path)

params_dict = {
    'x_threshold': [2.4, 2.1, 1.8, 1.5, 1.2]
}

params_sampler = SequentialParams(params_dict)
params_list = params_sampler.sample(NUM_TRAINING_ENVS) + [(params_sampler.sample1_means())]

pickle.dump(params_list, open(save_path + 'params_list.pkl', "wb"))


# params_path = './experiments__2020_12_30__00_23__3rd/params_list.pkl'
# params_list = pickle.load(open(params_path, 'rb'))


def train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size, subsave, iteration,
              last_round_no_mer, is_evolving, seed):
    training_timesteps = META_TRAINING_TIMESTEPS
    params = params_list
    if not is_evolving:
        params = [params[-1]]

    start_time = time()
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    final_eval_env = gym.make(env_name)
    final_parameters_dict = params_sampler.sample1_means()
    change_env_parameters(final_eval_env, parameter_dict=final_parameters_dict)
    tensorboard_path = subsave + '/tb_' + str(iteration)

    optimizer_kwargs = {}
    policy_kwargs = {
        'optimizer_class': th.optim.Adam,
        'optimizer_kwargs': optimizer_kwargs,
    }
    model = model_alg(MlpPolicy, env, verbose=0, buffer_size=buffer_size, batch_size=BATCH_SIZE,
                      learning_rate=LEARNING_RATE,
                      learning_starts=LEARNING_STARTS,
                      gradient_steps=GRADIENT_STEPS, policy_kwargs=policy_kwargs, mer_s=MER_S, mer_gamma=MER_GAMMA,
                      monitor_wrapper=True,
                      tensorboard_log=tensorboard_path,
                      reset_optimizers_during_training=reset_optimizers_every_iter,
                      seed=seed
                      )

    for i_param, param in enumerate(params):
        log_name = 'run_' + str(i_param)
        if i_param == (len(params) - 1):
            training_timesteps = FINAL_TRAINING_TIMESTEPS
            log_name += '_final'
        change_env_parameters(env, eval_env, parameter_dict=param)
        if model_alg.__name__ == 'SACMER' and last_round_no_mer and (i_param == (len(params) - 1)):
            is_reservoir = False
            is_mer = False
        else:  # This will not have any effect on regular SAC
            is_reservoir = True
            is_mer = True
        model.update_env(env, monitor_wrapper=False, is_reservoir=is_reservoir,
                         reset_optimizers=reset_optimizers_between_envs)  # environment already wrapped so
        # monitor_wrapper=False
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=None,
                                     log_path=tensorboard_path + '/' + log_name + '/running_eval',
                                     eval_freq=EVAL_FREQ,
                                     n_eval_episodes=N_EVAL_EPISODES,
                                     deterministic=True, render=False)
        if is_evolving:
            final_eval_callback = EvalCallback(final_eval_env,
                                               best_model_save_path=None,
                                               log_path=tensorboard_path + '/' + log_name + '/final_eval',
                                               eval_freq=EVAL_FREQ,
                                               n_eval_episodes=N_EVAL_EPISODES,
                                               deterministic=True, render=False)
        else:
            final_eval_callback = EventCallback()
        model.learn(total_timesteps=training_timesteps, log_interval=1, reset_num_timesteps=False,
                    tb_log_name=log_name, is_mer=is_mer, callback=CallbackList([eval_callback, final_eval_callback]))
        env.reset()
        eval_env.reset()
    if iteration == 0:  # saving models fills up storage, so we only save one (which we will also probably not use)
        model.save(subsave + 'model_' + str(iteration))
    print(f"Done. Total time = {time() - start_time} seconds.")


total_start_time = time()
################################################################
# SAC - no optimizer reset between environment updates
################################################################
source_subsave = save_path + 'SAC_no_reset/'
model_alg = SACExpanded
reset_optimizers_between_envs = False
reset_optimizers_every_iter = False
last_round_no_mer = False

for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'final_only/', i, last_round_no_mer,
                  False, seed)

################################################################
# SAC - with optimizer reset between environment updates
################################################################
source_subsave = save_path + 'SAC_with_reset/'
model_alg = SACExpanded
reset_optimizers_between_envs = True
reset_optimizers_every_iter = False
last_round_no_mer = False

for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter,
                  buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)
        # train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter,buffer_size,
        # subsave + 'final_only/', i, last_round_no_mer,
        #           False,seed)  # not necessary, this was already tested since there are no optimizer resets
        #           if only training on final env

################################################################
# SACMER - no changes - reset optimizer every run
################################################################
source_subsave = save_path + 'SACMER_no_end_standard_with_resets/'
model_alg = SACMER
reset_optimizers_between_envs = False
reset_optimizers_every_iter = True
last_round_no_mer = False
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'final_only/', i, last_round_no_mer,
                  False, seed)

################################################################
# SACMER - final training round is standard - reset optimizer every run
################################################################
source_subsave = save_path + 'SACMER_end_standard_with_resets/'
model_alg = SACMER
reset_optimizers_between_envs = False
reset_optimizers_every_iter = True
last_round_no_mer = True
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)

################################################################
# SACMER - no changes - no reset optimizers every run
################################################################
source_subsave = save_path + 'SACMER_no_end_standard/'
model_alg = SACMER
reset_optimizers_between_envs = False
reset_optimizers_every_iter = False
last_round_no_mer = False
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'final_only/', i, last_round_no_mer,
                  False, seed)

################################################################
# SACMER - final training round is standard - no reset optimizers every run
################################################################
source_subsave = save_path + 'SACMER_end_standard/'
model_alg = SACMER
reset_optimizers_between_envs = False
reset_optimizers_every_iter = False
last_round_no_mer = True
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        seed = initial_seed + i
        train_alg(model_alg, reset_optimizers_between_envs, reset_optimizers_every_iter, buffer_size,
                  subsave + 'evolving/', i, last_round_no_mer,
                  True, seed)

print(f'All done! Total time = {time() - total_start_time} seconds.')
