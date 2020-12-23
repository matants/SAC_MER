import gym
import numpy as np
import torch as th
from datetime import datetime
import subprocess
from time import time

from stable_baselines3 import SAC
from sac_reservoir import ReservoirSAC
from sac_mer import SACMER
from sac_expanded import SACExpanded
from stable_baselines3.sac import MlpPolicy
import gym_continuouscartpole  # not necessary to import but this checks if it is installed
from utils import change_env_parameters
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EventCallback

NUM_OF_REDOS = 10
EVAL_FREQ = 10
N_EVAL_EPISODES = 5

env_name = 'gym_continuouscartpole:ContinuousCartPole-v1'
model_algs = [ReservoirSAC, SACMER]
buffer_sizes = [4000, 1000, 100]

now = datetime.now().strftime("%Y_%m_%d__%H_%M")
save_path = './experiments__' + now + '/'


def train_alg(model_alg, reset_optimizers, buffer_size, subsave, iteration, last_round_no_mer, is_evolving):
    lengths = [0.5, 0.4, 0.3, 0.2]
    training_timesteps = 1000
    if not is_evolving:
        training_timesteps *= len(lengths)
        lengths = [lengths[-1]]

    start_time = time()
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    final_eval_env = gym.make(env_name)
    final_parameters_dict = {'length': 0.2}
    change_env_parameters(final_eval_env, parameter_dict=final_parameters_dict)
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

    for i_length, length in enumerate(lengths):
        parameter_dict = {'length': length}
        log_name = 'run_' + str(i_length) + '_len_' + str(length)
        change_env_parameters(env, eval_env, parameter_dict=parameter_dict)
        if model_alg.__name__ == 'SACMER' and last_round_no_mer and length == 0.2:
            is_reservoir = False
            is_mer = False
        else:  # This will not have any effect on regular SAC
            is_reservoir = True
            is_mer = True
        model.update_env(env, monitor_wrapper=False, is_reservoir=is_reservoir,
                         reset_optimizers=reset_optimizers)  # environment already wrapped so monitor_wrapper=False
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
# SACMER - no changes
################################################################
source_subsave = save_path + 'SACMER_no_end_standard/'
model_alg = SACMER
reset_optimizers = False
last_round_no_mer = False
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
                  is_evolving=True)
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
                  is_evolving=False)

################################################################
# SACMER - final training round is standard
################################################################
source_subsave = save_path + 'SACMER_end_standard/'
model_alg = SACMER
reset_optimizers = False
last_round_no_mer = True
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
                  is_evolving=True)
        # train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
        #           is_evolving=False)  # Not necessary, this is covered in SAC alone

################################################################
# SAC - no optimizer reset between environment updates
################################################################
source_subsave = save_path + 'SAC_no_reset/'
model_alg = SACExpanded
reset_optimizers = False
last_round_no_mer = False
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
                  is_evolving=True)
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
                  is_evolving=False)

################################################################
# SAC - with optimizer reset between environment updates
################################################################
source_subsave = save_path + 'SAC_with_reset/'
model_alg = SACExpanded
reset_optimizers = True
last_round_no_mer = False
for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
                  is_evolving=True)
        # train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
        #           is_evolving=False)  # not necessary, this was already tested since there are no optimizer resets if only training on final env

print(f'All done! Total time = {time() - total_start_time} seconds.')
