import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from collections import namedtuple
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch


def plot_single_run_results(pd_data):
    """

    :param npz_path:
    :return:
    """
    sns.lineplot(data=pd_data, x='timesteps', y='rewards')
    plt.grid()
    plt.show()
    db = 1


def convert_npz_to_dataframe(npz_path):
    data = np.load(npz_path)
    rewards = data.f.results.squeeze()
    if len(rewards.shape) == 1:
        rewards = np.expand_dims(rewards, axis=1)
    timesteps = data.f.timesteps
    cols = ['c' + str(i) for i in range(np.shape(rewards)[1])]
    cols_content = [rewards[:, i] for i in range(np.shape(rewards)[1])]
    d = {'timesteps': timesteps, **dict(zip(cols, cols_content))}
    pd_data = pd.DataFrame(d).melt(id_vars='timesteps', value_vars=cols, value_name='rewards')
    return pd_data


def merge_tbs__final_only(root_path):
    files_of_data = []
    for root, dirs, files in os.walk(root_path):
        if len(files) >= 1:
            if files[0].split('.')[-1] == 'npz':
                files_of_data.append(root + '/' + files[0])
    df_list = []
    for fname in files_of_data:
        df_list.append(convert_npz_to_dataframe(fname))
    if len(df_list) == 0:
        raise ValueError("No data found.")
    elif len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def merge_tbs__evolving(root_path, env_ind, is_final_eval):
    if is_final_eval:
        dname = 'final_eval'
    else:
        dname = 'running_eval'
    files_of_data = []
    for root, dirs, files in os.walk(root_path):
        if len(files) >= 1:
            if (files[0].split('.')[-1] == 'npz') and (f'run_{env_ind}' in root) and (dname in root):
                files_of_data.append(root + '/' + files[0])
    df_list = []
    for fname in files_of_data:
        df_list.append(convert_npz_to_dataframe(fname))
    if len(df_list) == 0:
        raise ValueError("No data found.")
    elif len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def merge_tbs__evolving__all_envs_together(root_path, is_final_eval):
    if is_final_eval:
        dname = 'final_eval'
    else:
        dname = 'running_eval'
    files_of_data = []
    for root, dirs, files in os.walk(root_path):
        if len(files) >= 1:
            if (files[0].split('.')[-1] == 'npz') and (dname in root):
                files_of_data.append(root + '/' + files[0])
    df_list = []
    for fname in files_of_data:
        df_list.append(convert_npz_to_dataframe(fname))
    if len(df_list) == 0:
        raise ValueError("No data found.")
    elif len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


if __name__ == '__main__':
    root_path = 'C:/Users/matan/Documents/SAC_MER/experiments__2021_02_27__14_18_dqn_lengths_longtrain/'
    # ############################################################################################
    # # Comparing final_only training runs between algorithms
    # ############################################################################################
    # algorithms_dirs = ['SAC_no_reset', 'SACMER_no_end_standard', 'SACMER_no_end_standard_with_resets']
    # algorithms_names = ['SAC', 'SAC + MER', 'SAC + MER with optimizer resets']
    # buffer_sizes = [50000]
    # for buffer in buffer_sizes:
    #     df_arr = []
    #     for i_alg, alg in enumerate(algorithms_dirs):
    #         path = root_path + "/" + alg + f'/buffer_{buffer}/final_only'
    #         df = merge_tbs__final_only(path)
    #         df_arr.append(df)
    #         df = df.sort_values(by=['timesteps'])
    #         df_avg = df.rolling(100, on='timesteps')
    #         sns.lineplot(data=df, x='timesteps', y='rewards')
    #         # sns.lineplot(data=df_avg, x='timesteps', y='rewards')
    #     plt.legend(algorithms_names)
    #     plt.suptitle(f'Buffer size = {buffer}')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Reward')
    #     plt.axhline(y=500, linestyle='--', color='black')
    #     plt.grid()
    #     plt.show()
    #
    # ############################################################################################
    # # Comparing evolving running_eval_between all algorithms
    # ############################################################################################
    # algorithms_dirs = ['SAC_no_reset', 'SAC_with_reset', 'SACMER_no_end_standard',
    #                    'SACMER_no_end_standard_with_resets', 'SACMER_end_standard', 'SACMER_end_standard_with_resets']
    # algorithms_names = ['SAC (without optimizer resets)', 'SAC (with optimizer resets between envs)', 'SAC + MER',
    #                     'SAC + MER with optimizer resets',
    #                     'SAC + MER (final env regular SAC)', 'SAC + MER with optimizer resets (final env regular
    #                     SAC)']
    # buffer_sizes = [50000]  # , 5000, 256]
    # env_switch_times = [10000, 20000, 30000, 40000]
    # for buffer in buffer_sizes:
    #     df_arr = []
    #     for i_alg, alg in enumerate(algorithms_dirs):
    #         path = root_path + alg + f'/buffer_{buffer}/evolving'
    #         df = merge_tbs__evolving__all_envs_together(path, is_final_eval=False)
    #         df = df.sort_values(by=['timesteps'])
    #         df_arr.append(df)
    #         sns.lineplot(data=df, x='timesteps', y='rewards')
    #     plt.legend(algorithms_names)
    #     plt.suptitle(f'Buffer size = {buffer}')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Reward')
    #     for x in env_switch_times:
    #         plt.axvline(x=x, linestyle='--', color='black')
    #     plt.axhline(y=500, linestyle='--', color='black')
    #     plt.grid()
    #     plt.show()

    ############################################################################################
    # Comparing final_only training runs between algorithms
    ############################################################################################
    algorithms_dirs = ['DQN_no_reset', 'DQNMER_no_end_standard']
    algorithms_names = ['DQN', 'DQN + MER']
    buffer_sizes = [400000]#, 5000, 256]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + "/" + alg + f'/buffer_{buffer}/final_only'
            df = merge_tbs__final_only(path)
            df_arr.append(df)
            df = df.sort_values(by=['timesteps'])
            df_avg = df.rolling(100, on='timesteps')
            sns.lineplot(data=df, x='timesteps', y='rewards')
            # sns.lineplot(data=df_avg, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.axhline(y=500, linestyle='--', color='black')
        plt.grid()
        plt.show()

    ############################################################################################
    # Comparing evolving running_eval_between all algorithms
    ############################################################################################
    algorithms_dirs = ['DQN_no_reset', 'DQN_with_reset', 'DQNMER_no_end_standard',
                       'DQNMER_end_standard']
    algorithms_names = ['DQN (without optimizer resets)', 'DQN (with optimizer resets between envs)', 'DQN + MER',
                        'DQN + MER (final env regular DQN)']
    buffer_sizes = [400000] #, 5000, 256]
    env_switch_times = [100000, 200000, 300000]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + alg + f'/buffer_{buffer}/evolving'
            df = merge_tbs__evolving__all_envs_together(path, is_final_eval=False)
            df = df.sort_values(by=['timesteps'])
            df_arr.append(df)
            sns.lineplot(data=df, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        for x in env_switch_times:
            plt.axvline(x=x, linestyle='--', color='black')
        plt.axhline(y=500, linestyle='--', color='black')
        plt.grid()
        plt.show()
