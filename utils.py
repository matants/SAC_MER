import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback


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


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    parameter_dict = {'length': 1}
    change_env_parameters(env, parameter_dict=parameter_dict)
    print(env)
