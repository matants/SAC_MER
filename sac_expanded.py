from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from stable_baselines3 import SAC
from stable_baselines3.common.base_class import maybe_make_env


class SACExpanded(SAC):
    # adding kwargs to __init__ to handle kwargs not used by SAC
    def __init__(
            self,
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable] = 3e-4,
            buffer_size: int = int(1e6),
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int = 1,
            gradient_steps: int = 1,
            n_episodes_rollout: int = -1,
            action_noise: Optional[ActionNoise] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Dict[str, Any] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            **kwargs
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def update_env(self, env, support_multi_env: bool = False, create_eval_env: bool = False,
                   monitor_wrapper: bool = True, ):
        """
        Replace current env with new env.
        :param env:
        :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
        :param create_eval_env: Whether to create a second environment that will be
            used for evaluating the agent periodically. (Only available when passing string for the environment)
        :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
        :return:
        """
        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    self.eval_env = maybe_make_env(env, monitor_wrapper, self.verbose)

            env = maybe_make_env(env, monitor_wrapper, self.verbose)
            env = self._wrap_env(env, self.verbose)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )