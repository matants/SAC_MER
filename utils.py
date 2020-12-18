import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    data = np.load(npz_path)
    ep_lengths = data.f.ep_lengths
    timesteps = data.f.timesteps
    cols = ['c' + str(i) for i in range(np.shape(ep_lengths)[1])]
    cols_content = [ep_lengths[:, i] for i in range(np.shape(ep_lengths)[1])]
    d = {'timesteps': timesteps, **dict(zip(cols, cols_content))}
    pd_data = pd.DataFrame(d).melt(id_vars='timesteps', value_vars=cols, value_name='ep_lengths')

    sns.lineplot(data=pd_data, x='timesteps', y='ep_lengths')
    plt.grid()
    plt.show()
    db = 1


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    # parameter_dict = {'length': 1}
    # change_env_parameters(env, parameter_dict=parameter_dict)
    # print(env)

    npz_path = './run_1_len_0.1/evaluations.npz'
    plot_single_run_results(npz_path)
