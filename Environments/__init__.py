import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ContinuousCartPole-v0',
    entry_point='Environments.envs.continuous_cartpole_env:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='ContinuousCartPole-v1',
    entry_point='Environments.envs.continuous_cartpole_env:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)
