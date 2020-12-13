from stable_baselines3 import SAC
from stable_baselines3.common.base_class import maybe_make_env


class SACExpanded(SAC):
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