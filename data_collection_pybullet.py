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
from sac_mer_variations import SACMER_Q, SACMER_P, SACMER_T
from stable_baselines3.sac import MlpPolicy
# import gym_continuouscartpole  # not necessary to import but this checks if it is installed
import toy_navigation_envs
import pybullet
import pybulletgym
from utils import change_env_parameters, Param, AlternatingParamsUniform, AlternatingParamsOnCircle, \
    AlternatingParamsSemiCircleBot
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EventCallback
import pickle
import os

random.seed(685475328)
NUM_OF_REDOS = 80  # how many times we run the training loops (for confidence bounds)
EVAL_FREQ = 5000
N_EVAL_EPISODES = 1
NUM_TRAINING_ENVS = 10
MER_S = 2
MER_GAMMA = 0.5
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
LEARNING_STARTS = 512
GRADIENT_STEPS = 4
META_TRAINING_TIMESTEPS = 10000
FINAL_TRAINING_TIMESTEPS = 50000

env_name = "toy_navigation_envs:PointRobotSparse-v0"
# env_name = "toy_navigation_envs:PointRobot-v0"
buffer_sizes = [50000]  # , 10000]

now = datetime.now().strftime("%Y_%m_%d__%H_%M")
save_path = './experiments__' + now + '/'
os.mkdir(save_path)

params_dict = {
    '_goal': np.array([0.46010313, 0.88786548]),
    'goals': [np.array([0.46010313, 0.88786548])],
    '_state': np.array([0, 0]),
    'modify_init_state_dist': False
}

params_sampler = AlternatingParamsSemiCircleBot(params_dict)
# params_list = params_sampler.sample(NUM_TRAINING_ENVS) + [(params_sampler.sample1_means())]
params_list = params_sampler.sample(NUM_OF_REDOS)
pickle.dump(params_list, open(save_path + 'params_list.pkl', "wb"))


# params_path = './experiments__2021_01_09__11_14/params_list.pkl'
# params_list = pickle.load(open(params_path, 'rb'))


def train_alg(model_alg, reset_optimizers, buffer_size, subsave, iteration, last_round_no_mer, is_evolving,
              gradient_steps=GRADIENT_STEPS, params_list=params_list):
    training_timesteps = META_TRAINING_TIMESTEPS
    params = params_list
    if not is_evolving:
        params = [params[-1]]

    start_time = time()
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    final_eval_env = gym.make(env_name)
    final_parameters_dict = params[-1]
    change_env_parameters(final_eval_env, parameter_dict=final_parameters_dict)
    tensorboard_path = subsave + '/tb_' + str(iteration)

    optimizer_kwargs = {}
    policy_kwargs = {
        'optimizer_class': th.optim.Adam,
        'optimizer_kwargs': optimizer_kwargs,
    }
    model = model_alg(MlpPolicy, env, verbose=1, buffer_size=buffer_size, batch_size=BATCH_SIZE,
                      learning_rate=LEARNING_RATE,
                      learning_starts=LEARNING_STARTS,
                      gradient_steps=gradient_steps, policy_kwargs=policy_kwargs, mer_s=MER_S, mer_gamma=MER_GAMMA,
                      monitor_wrapper=True,
                      tensorboard_log=tensorboard_path)

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
            final_eval_callback = EventCallback()  # empty callback
        model.learn(total_timesteps=training_timesteps, log_interval=1, reset_num_timesteps=False,
                    tb_log_name=log_name, is_mer=is_mer, callback=CallbackList([eval_callback, final_eval_callback]))
        env.reset()
        eval_env.reset()
    # if iteration == 0:  # saving models fills up storage, so we only save one (which we will also probably not use)
    model.save(subsave + 'model_' + str(iteration), include=['replay_buffer'])
    print(f"Done. Total time = {time() - start_time} seconds.")


total_start_time = time()

###############################################################
# SAC - no optimizer reset between environment updates
###############################################################
source_subsave = save_path + 'SAC_no_reset/'
model_alg = SACExpanded
reset_optimizers = False
last_round_no_mer = False
gradient_steps = GRADIENT_STEPS

for buffer_size in buffer_sizes:
    subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
    for i in range(NUM_OF_REDOS):
        params = [params_list[i]]
        # train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
        #           is_evolving=True, gradient_steps=gradient_steps)
        train_alg(model_alg, reset_optimizers, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
                  is_evolving=False, params_list=params)
#
# ################################################################
# # SAC - with optimizer reset between environment updates
# ################################################################
# source_subsave = save_path + 'SAC_with_reset/'
# model_alg = SACExpanded
# reset_optimizers_between_envs = True
# last_round_no_mer = False
# gradient_steps = GRADIENT_STEPS
#
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(3, 3 + NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True, gradient_steps=gradient_steps)
#         # train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#         #           is_evolving=False)  # not necessary, this was already tested since there are no optimizer resets
#         #           if only training on final env
#
# ################################################################
# # SACMER - no changes
# ################################################################
# source_subsave = save_path + 'SACMER_no_end_standard/'
# model_alg = SACMER
# reset_optimizers_between_envs = False
# last_round_no_mer = False
# gradient_steps = GRADIENT_STEPS + 1
#
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(3, 3 + NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True, gradient_steps=gradient_steps)
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#                   is_evolving=False)
#
# ################################################################
# # SACMER - final training round is standard
# ################################################################
# source_subsave = save_path + 'SACMER_end_standard/'
# model_alg = SACMER
# reset_optimizers_between_envs = False
# last_round_no_mer = True
# gradient_steps = GRADIENT_STEPS + 1
#
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True, gradient_steps=gradient_steps)
#         # train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#         #           is_evolving=False)  # Not necessary, this is covered in SAC alone

# ################################################################
# # SACMER_P - no changes
# ################################################################
# source_subsave = save_path + 'SACMER_P_no_end_standard/'
# model_alg = SACMER_P
# reset_optimizers_between_envs = False
# last_round_no_mer = False
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True)
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#                   is_evolving=False)
#
# ################################################################
# # SACMER_P - final training round is standard
# ################################################################
# source_subsave = save_path + 'SACMER_P_end_standard/'
# model_alg = SACMER_P
# reset_optimizers_between_envs = False
# last_round_no_mer = True
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True)
#         # train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#         #           is_evolving=False)  # Not necessary, this is covered in SAC alone
#
# ################################################################
# # SACMER_T - no changes
# ################################################################
# source_subsave = save_path + 'SACMER_T_no_end_standard/'
# model_alg = SACMER_T
# reset_optimizers_between_envs = False
# last_round_no_mer = False
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True)
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#                   is_evolving=False)
#
# ################################################################
# # SACMER_T - final training round is standard
# ################################################################
# source_subsave = save_path + 'SACMER_T_end_standard/'
# model_alg = SACMER_T
# reset_optimizers_between_envs = False
# last_round_no_mer = True
# for buffer_size in buffer_sizes:
#     subsave = source_subsave + 'buffer_' + str(buffer_size) + '/'
#     for i in range(NUM_OF_REDOS):
#         train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'evolving/', i, last_round_no_mer,
#                   is_evolving=True)
#         # train_alg(model_alg, reset_optimizers_between_envs, buffer_size, subsave + 'final_only/', i, last_round_no_mer,
#         #           is_evolving=False)  # Not necessary, this is covered in SAC alone


print(f'All done! Total time = {time() - total_start_time} seconds.')
