from pettingzoo.utils import wrappers
from .env import CatanAECEnv
from .actions import normalize_action


def env(**kwargs):
    """Factory function for creating the AEC environment."""
    env_instance = CatanAECEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def parallel_env(**kwargs):
    """Factory function for creating the parallel API version of the environment."""
    from pettingzoo.utils import aec_to_parallel

    aec_env_instance = env(**kwargs)
    parallel_env_instance = aec_to_parallel(aec_env_instance)
    return parallel_env_instance
