import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def plot_single_run_results(npz_path):
    """

    :param npz_path:
    :return:
    """
    pd_data = convert_npz_to_dataframe(npz_path)
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


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    # parameter_dict = {'length': 1}
    # change_env_parameters(env, parameter_dict=parameter_dict)
    # print(env)

    npz_path = './SEE_IF_RUN_experiments__2020_12_19__22_09/SAC_no_reset/final_only/tb_0/run_0_len_0.2/running_eval/evaluations.npz'
    plot_single_run_results(npz_path)
