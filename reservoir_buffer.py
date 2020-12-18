from stable_baselines3.common.buffers import ReplayBuffer
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from random import randint

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ReservoirBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage)
        self.is_reservoir = True

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
            experience_index: int) -> None:
        if not self.full or not self.is_reservoir:
            super().add(obs, next_obs, action, reward, done)
        else:
            pos = randint(0, experience_index)
            if pos < self.buffer_size:
                self.pos = pos
                # Copy to avoid modification by reference
                self.observations[self.pos] = np.array(obs).copy()
                if self.optimize_memory_usage:
                    self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
                else:
                    self.next_observations[self.pos] = np.array(next_obs).copy()

                self.actions[self.pos] = np.array(action).copy()
                self.rewards[self.pos] = np.array(reward).copy()
                self.dones[self.pos] = np.array(done).copy()
