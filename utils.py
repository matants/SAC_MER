import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from collections import namedtuple
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os
import torch

Param = namedtuple('Param', ['mean', 'half_range'])


def seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


class AlternatingParamsUniform:
    def __init__(self, params_dict):
        self.params_dict = params_dict

    def sample1(self):
        ret_dict = {}
        for key in self.params_dict:
            param = self.params_dict[key]
            param_val = (random.random() - 0.5) * 2 * param.half_range + param.mean
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


class AlternatingParamsOnCircle:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.radius = np.sqrt(np.sum([params_dict[i] ** 2 for i in params_dict]))

    def sample1(self):
        ret_dict = {}
        angle = random() * 2 * np.pi
        for i_key, key in enumerate(self.params_dict):
            if i_key == 0:
                ret_dict[key] = self.radius * np.cos(angle)
            if i_key == 1:
                ret_dict[key] = self.radius * np.sin(angle)
        return ret_dict

    def sample(self, n):
        return [self.sample1() for _ in range(n)]

    def sample1_means(self):
        return self.params_dict


class AlternatingParamsSemiCircleBot:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.radius = np.linalg.norm(params_dict['_goal'])

    def sample1(self):
        ret_dict = {}
        angle = random() * np.pi
        ret_dict['_goal'] = np.array([self.radius * np.cos(angle), self.radius * np.sin(angle)])
        ret_dict['goals'] = [np.array([self.radius * np.cos(angle), self.radius * np.sin(angle)])]
        ret_dict['_state'] = np.array([0, 0])
        ret_dict['modify_init_state_dist'] = False
        return ret_dict

    def sample(self, n):
        return [self.sample1() for _ in range(n)]

    def sample1_means(self):
        return self.params_dict


class SequentialParams:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.index = 0
        lens = []
        for key in params_dict:
            lens.append(len(params_dict[key]))
        if lens.count(lens[0]) != len(lens):
            raise ValueError("Lengths don't match.")
        self.len = lens[0]

    def sample1(self):
        ret_dict = {}
        for key in self.params_dict:
            ret_dict[key] = self.params_dict[key][self.index]
        self.index += 1
        return ret_dict

    def sample(self, n):
        return [self.sample1() for _ in range(n)]

    def sample1_means(self):
        ret_dict = {}
        for key in self.params_dict:
            ret_dict[key] = self.params_dict[key][-1]
        return ret_dict


def change_env_parameters(env: GymEnv, eval_env: Optional[GymEnv] = None, parameter_dict: Dict = {}):
    '''

    :param env:
    :param eval_env:
    :param parameter_dict:
    :return:
    '''
    for parameter in parameter_dict:
        try:
            where_to_change = env.env
        except AttributeError:
            where_to_change = env
        setattr(where_to_change, parameter, parameter_dict[parameter])
        # assert getattr(where_to_change, parameter) == parameter_dict[parameter], "Set attribute failed!"
        if eval_env is not None:
            try:
                where_to_change = eval_env.env
            except AttributeError:
                where_to_change = eval_env
            setattr(where_to_change, parameter, parameter_dict[parameter])
            # assert getattr(where_to_change, parameter) == parameter_dict[parameter], "Set attribute failed!"

