# TODO: consider deleting — deprecated, unimported, excluded from wheel.
# import numpy as np
# from typing import Any, Tuple, Dict, SupportsFloat as float_t
# import gymnasium as gym
# from gymnasium.core import Wrapper
# from pettingzoo.utils.wrappers.base import BaseWrapper
# from pettingzoo.utils.env import ActionType, AgentID, ObsType, AECEnv


# class InitialMovesWrapper(BaseWrapper):
#     def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], initial_moves: list):
#         super().__init__(env)
#         self.initial_moves = initial_moves

#     def reset(self, seed: int | None = None, options: dict | None = None):
#         self.env.reset(seed=seed, options=options)
#         for move in self.initial_moves:
#             self.step(move)

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         obs = self.observe(self.env.agent_selection) if observe else None
#         return obs, reward, term, trunc, info

#     def state(self) -> np.ndarray:
#         return self.env.state()


# class CatanatronWrapper(Wrapper):
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self.old_key, self.new_key = "valid_actions", "legal_moves"

#     def step(self, action: Any) -> Tuple[Any, float_t, bool, bool, Dict[str, Any]]:
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         if self.old_key in info:
#             info[self.new_key] = info.pop(self.old_key)
#         return obs, reward, terminated, truncated, info

#     def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
#         obs, info = self.env.reset(**kwargs)
#         if self.old_key in info:
#             info[self.new_key] = info.pop(self.old_key)
#         return obs, info
