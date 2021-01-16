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
from copy import deepcopy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer


class SACExpanded(SAC):
    # adding kwargs to __init__ to handle kwargs not used by SAC
    def __init__(
            self,
            policy: Union[str, Type[SACPolicy]],
            env: GymEnv,
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
            monitor_wrapper: bool = True,
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
        self.update_env(env, support_multi_env=False, create_eval_env=create_eval_env, monitor_wrapper=monitor_wrapper,
                        reset_optimizers=False)

    def update_env(self, env, support_multi_env: bool = False,
                   eval_env: Optional[GymEnv] = None, monitor_wrapper: bool = True, reset_optimizers: bool = False,
                   **kwargs):
        """
        Replace current env with new env.
        :param env: Gym environment (activated, not a string).
        :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
        :param eval_env: Environment to use for evaluation (optional).
        :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
        :param reset_optimizers: Whether to reset optimizers (momentums, etc.).
        :param kwargs: Does nothing, just so more arguments can pass without method failing
        :return:
        """
        if reset_optimizers:
            optimizers = []
            if self.actor is not None:
                optimizers.append(self.actor.optimizer)
            if self.critic is not None:
                optimizers.append(self.critic.optimizer)
            if self.ent_coef_optimizer is not None:
                optimizers.append(self.ent_coef_optimizer)

            # Reset optimizers:
            for i_optimizer, optimizer in enumerate(optimizers):
                optimizer.__init__(optimizer.param_groups[0]['params'])
                optimizers[i_optimizer] = optimizer

        if env is not None:
            if eval_env is not None:
                self.eval_env = eval_env
                if monitor_wrapper:
                    self.eval_env = Monitor(self.eval_env, filename=None)

            if monitor_wrapper:
                env = Monitor(env, filename=None)
            env = self._wrap_env(env, self.verbose)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "SAC",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            **kwargs
    ) -> OffPolicyAlgorithm:
        """

        :param total_timesteps:
        :param callback:
        :param log_interval:
        :param eval_env:
        :param eval_freq:
        :param n_eval_episodes:
        :param tb_log_name:
        :param eval_log_path:
        :param reset_num_timesteps:
        :param kwargs: Does nothing just so method doesn't break.
        :return:
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def add_memories_from_another_replay_mem(self, another_replay_mem: ReplayBuffer):
        for i in range(another_replay_mem.buffer_size):
            self.replay_buffer.add(
                obs=another_replay_mem.observations[i],
                next_obs=another_replay_mem.next_observations[i],
                action=another_replay_mem.actions[i],
                reward=another_replay_mem.rewards[i],
                done=another_replay_mem.dones[i]
            )
