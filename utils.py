import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from collections import namedtuple
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random
import os

Param = namedtuple('Param', ['mean', 'half_range'])


class AlternatingParamsUniform:
    def __init__(self, params_dict):
        self.params_dict = params_dict

    def sample1(self):
        ret_dict = {}
        for key in self.params_dict:
            param = self.params_dict[key]
            param_val = (random() - 0.5) * 2 * param.half_range + param.mean
            ret_dict[key] = param_val
        return ret_dict

    def sample(self, n):
        return [self.sample1() for _ in range(n)]

    def sample1_means(self):
        ret_dict = {}
        for key in self.params_dict:
            param = self.params_dict[key]
            ret_dict[key] = param.mean
        return ret_dict


def change_env_parameters(env: GymEnv, eval_env: Optional[GymEnv] = None, parameter_dict: Dict = {}):
    '''

    :param env:
    :param eval_env:
    :param parameter_dict:
    :return:
    '''
    for parameter in parameter_dict:
        setattr(env.env, parameter, parameter_dict[parameter])
        assert getattr(env.env, parameter) == parameter_dict[parameter], "Set attribute failed!"
        if eval_env is not None:
            setattr(eval_env.env, parameter, parameter_dict[parameter])
            assert getattr(eval_env.env, parameter) == parameter_dict[parameter], "Set attribute failed!"


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
    # env = gym.make('CartPole-v0')
    # parameter_dict = {'length': 1}
    # change_env_parameters(env, parameter_dict=parameter_dict)
    # print(env)

    # npz_path = './SEE_IF_RUN_experiments__2020_12_19__22_09/SAC_no_reset/final_only/tb_0/run_0_len_0.2/running_eval
    # /evaluations.npz'
    # plot_single_run_results(convert_npz_to_dataframe(npz_path))

    # final_only_path = 'C:/Users/matan/Documents/SAC_MER/experiments__2020_12_20__23_37/SAC_no_reset/buffer_4000
    # /final_only'
    # df = merge_tbs__final_only(final_only_path)
    # plot_single_run_results(df)

    # evolving_path = 'C:/Users/matan/Documents/SAC_MER/experiments__2020_12_20__23_37/SAC_no_reset/buffer_4000
    # /evolving'
    # df_final_eval = merge_tbs__evolving(evolving_path, 2, True)
    # df_running_eval = merge_tbs__evolving(evolving_path, 2, False)
    # sns.lineplot(data=df_final_eval, x='timesteps', y='rewards')
    # sns.lineplot(data=df_running_eval, x='timesteps', y='rewards')
    # plt.legend(['final eval', 'running eval'])
    # plt.show()

    root_path = 'C:/Users/matan/Documents/SAC_MER/experiments__2021_01_02__19_01/'
    NUM_ENVS = 11
    ############################################################################################
    # Comparing final_only training runs between algorithms (mer shouldn't be helpful, but maybe with different batch
    # sizes? nah)
    ############################################################################################
    algorithms_dirs = ['SAC_no_reset', 'SACMER_T_no_end_standard']
    algorithms_names = ['SAC', 'SAC + MER']
    buffer_sizes = [30000]  # , 5000, 256]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + alg + f'/buffer_{buffer}/final_only'
            df = merge_tbs__final_only(path)
            df_arr.append(df)
            sns.lineplot(data=df, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.axhline(y=500)
        plt.grid()
        plt.show()

    ############################################################################################
    # Comparing evolving running_eval_between all algorithms
    ############################################################################################
    algorithms_dirs = ['SAC_no_reset', 'SAC_with_reset', 'SACMER_T_no_end_standard', 'SACMER_T_end_standard']
    algorithms_names = ['SAC (without optimizer resets)', 'SAC (with optimizer resets between envs)', 'SAC + MER',
                        'SAC + MER (final env regular SAC)']
    buffer_sizes = [30000]  # , 5000, 256]
    env_switch_times = []  # 10000, 20000, 30000]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + alg + f'/buffer_{buffer}/evolving'
            df = merge_tbs__evolving__all_envs_together(path, is_final_eval=False)
            df_arr.append(df)
            sns.lineplot(data=df, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        for x in env_switch_times:
            plt.axvline(x=x)
        plt.axhline(y=500)
        plt.grid()
        plt.show()

    ############################################################################################
    # Comparing final_only training runs between algorithms (mer shouldn't be helpful, but maybe with different batch
    # sizes? nah)
    ############################################################################################
    algorithms_dirs = ['SACMER_no_end_standard', 'SACMER_T_no_end_standard']
    algorithms_names = ['SACMER', 'SACMER_T']
    buffer_sizes = [30000]  # , 5000, 256]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + alg + f'/buffer_{buffer}/final_only'
            df = merge_tbs__final_only(path)
            df_arr.append(df)
            sns.lineplot(data=df, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.axhline(y=500)
        plt.grid()
        plt.show()

    ############################################################################################
    # Comparing evolving running_eval_between all algorithms
    ############################################################################################
    algorithms_dirs = ['SACMER_no_end_standard', 'SACMER_end_standard', 'SACMER_T_no_end_standard',
                       'SACMER_T_end_standard']
    algorithms_names = ['SAC + MER',
                        'SAC + MER (final env regular SAC)',
                        'SAC + MER_T',
                        'SAC + MER_T (final env regular SAC)',
                        ]
    buffer_sizes = [30000]  # , 5000, 256]
    env_switch_times = []  # 10000, 20000, 30000]
    for buffer in buffer_sizes:
        df_arr = []
        for i_alg, alg in enumerate(algorithms_dirs):
            path = root_path + alg + f'/buffer_{buffer}/evolving'
            df = merge_tbs__evolving__all_envs_together(path, is_final_eval=False)
            df_arr.append(df)
            sns.lineplot(data=df, x='timesteps', y='rewards')
        plt.legend(algorithms_names)
        plt.suptitle(f'Buffer size = {buffer}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        for x in env_switch_times:
            plt.axvline(x=x)
        plt.axhline(y=500)
        plt.grid()
        plt.show()
