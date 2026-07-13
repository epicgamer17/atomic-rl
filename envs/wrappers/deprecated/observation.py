# from __future__ import annotations
# from typing import Any
# from collections import deque
# import numpy as np
# import gymnasium as gym
# import gymnasium.spaces
# from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
# from pettingzoo.utils.wrappers.base import BaseWrapper


# class ChannelLastToFirstWrapper(BaseWrapper):
#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         orig_space = self.env.observation_space(agent)
#         if isinstance(orig_space, gymnasium.spaces.Box) and len(orig_space.shape) == 3:
#             h, w, c = orig_space.shape
#             return gymnasium.spaces.Box(
#                 low=orig_space.low.min(),
#                 high=orig_space.high.max(),
#                 shape=(c, h, w),
#                 dtype=orig_space.dtype,
#             )
#         elif (
#             isinstance(orig_space, gymnasium.spaces.Dict)
#             and "observation" in orig_space.spaces
#         ):
#             obs_space = orig_space["observation"]
#             if len(obs_space.shape) == 3:
#                 h, w, c = obs_space.shape
#                 return gymnasium.spaces.Box(
#                     low=obs_space.low.min(),
#                     high=obs_space.high.max(),
#                     shape=(c, h, w),
#                     dtype=obs_space.dtype,
#                 )
#         return orig_space

#     def observe(self, agent: AgentID) -> np.ndarray:
#         obs = self.env.observe(agent)
#         if isinstance(obs, dict) and "observation" in obs:
#             obs = obs["observation"]
#         if isinstance(obs, np.ndarray) and obs.ndim == 3:
#             obs = np.transpose(obs, (2, 0, 1))
#         return obs

#     def step(self, action: ActionType) -> None:
#         self.env.step(action)

#     def reset(self, seed: int | None = None, options: dict | None = None):
#         self.env.reset(seed=seed, options=options)

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         if observe and not term and not trunc:
#             obs = self.observe(self.env.agent_selection)
#         else:
#             obs = None
#         return obs, reward, term, trunc, info

#     def state(self) -> np.ndarray:
#         return self.env.state()


# class AppendAgentSelectionWrapper(BaseWrapper):
#     def observation_space(self, agent: AgentID) -> gym.spaces.Space:
#         orig_space = self.env.observation_space(agent)

#         def _append_box(obs_space: gym.spaces.Box):
#             if len(obs_space.shape) != 1:
#                 return obs_space
#             orig_low = np.asarray(obs_space.low).reshape(-1)
#             orig_high = np.asarray(obs_space.high).reshape(-1)
#             num_agents = len(self.env.possible_agents)
#             if num_agents == 0:
#                 return obs_space
#             appended_low = np.zeros((num_agents,), dtype=orig_low.dtype)
#             appended_high = np.ones((num_agents,), dtype=orig_high.dtype)
#             return gym.spaces.Box(
#                 low=np.concatenate([orig_low, appended_low]),
#                 high=np.concatenate([orig_high, appended_high]),
#                 shape=(obs_space.shape[0] + num_agents,),
#                 dtype=obs_space.dtype,
#             )

#         if (
#             isinstance(orig_space, gym.spaces.Dict)
#             and "observation" in orig_space.spaces
#         ):
#             return _append_box(orig_space["observation"])
#         elif isinstance(orig_space, gym.spaces.Box):
#             return _append_box(orig_space)
#         return orig_space

#     def observe(self, agent: AgentID) -> np.ndarray:
#         obs = self.env.observe(agent)
#         if isinstance(obs, dict) and "observation" in obs:
#             obs = obs["observation"]
#         obs = np.asarray(obs)
#         if obs.ndim != 1:
#             return obs
#         possible_agents = list(self.env.possible_agents)
#         num_agents = len(possible_agents)
#         if num_agents == 0:
#             return obs
#         try:
#             sel_agent = self.env.agent_selection
#             selected_index = possible_agents.index(sel_agent)
#         except Exception:
#             selected_index = None
#         oh_dtype = (
#             obs.dtype
#             if np.issubdtype(obs.dtype, (np.integer, np.floating))
#             else np.float32
#         )
#         one_hot = np.zeros((num_agents,), dtype=oh_dtype)
#         if selected_index is not None and 0 <= selected_index < num_agents:
#             one_hot[selected_index] = 1
#         return np.concatenate([obs, one_hot], axis=0)

#     def reset(self, seed: int | None = None, options: dict | None = None):
#         return self.env.reset(seed=seed, options=options)

#     def step(self, action: ActionType) -> None:
#         return self.env.step(action)

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         if observe and not term and not trunc:
#             obs = self.observe(self.env.agent_selection)
#         else:
#             obs = None
#         return obs, reward, term, trunc, info

#     def state(self) -> np.ndarray:
#         return self.env.state()


# class TwoPlayerPlayerPlaneWrapper(BaseWrapper):
#     def __init__(
#         self, env: AECEnv[AgentID, ObsType, ActionType], channel_first: bool = True
#     ):
#         super().__init__(env)
#         self.channel_first = channel_first

#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         orig_space = self.env.observation_space(agent)
#         if isinstance(orig_space, (gymnasium.spaces.Box, gymnasium.spaces.Dict)):
#             obs_space = (
#                 orig_space["observation"]
#                 if isinstance(orig_space, gymnasium.spaces.Dict)
#                 else orig_space
#             )
#             shape = obs_space.shape
#             if len(shape) == 3:
#                 if self.channel_first:
#                     c, h, w = shape
#                     new_shape = (c + 1, h, w)
#                 else:
#                     h, w, c = shape
#                     new_shape = (h, w, c + 1)
#                 return gymnasium.spaces.Box(
#                     low=0, high=1, shape=new_shape, dtype=obs_space.dtype
#                 )
#         return orig_space

#     def observe(self, agent: AgentID) -> np.ndarray:
#         obs = self.env.observe(agent)
#         if isinstance(obs, dict) and "observation" in obs:
#             obs = obs["observation"]
#         plane_val = 0 if agent == self.env.possible_agents[0] else 1
#         if isinstance(obs, np.ndarray) and obs.ndim == 3:
#             if self.channel_first:
#                 h, w = obs.shape[1], obs.shape[2]
#                 plane = np.full((1, h, w), plane_val, dtype=obs.dtype)
#                 obs = np.concatenate([obs, plane], axis=0)
#             else:
#                 h, w = obs.shape[0], obs.shape[1]
#                 plane = np.full((h, w, 1), plane_val, dtype=obs.dtype)
#                 obs = np.concatenate([obs, plane], axis=2)
#         return obs

#     def step(self, action: ActionType) -> None:
#         self.env.step(action)

#     def reset(self, seed: int | None = None, options: dict | None = None):
#         self.env.reset(seed=seed, options=options)

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         if observe and not term and not trunc:
#             obs = self.observe(self.env.agent_selection)
#         else:
#             obs = None
#         return obs, reward, term, trunc, info

#     def state(self) -> np.ndarray:
#         return self.env.state()


# class FrameStackWrapper(BaseWrapper):
#     def __init__(
#         self,
#         env: AECEnv[AgentID, ObsType, ActionType],
#         k: int = 4,
#         channel_first: bool = True,
#     ):
#         super().__init__(env)
#         self.k = k
#         self.channel_first = channel_first
#         self.stacks: dict[AgentID, deque] = {
#             agent: deque(maxlen=k) for agent in self.env.possible_agents
#         }

#     def reset(self, seed: int | None = None, options: dict | None = None):
#         self.env.reset(seed=seed, options=options)
#         for agent in self.env.agents:
#             self.stacks[agent].clear()
#             obs = self.env.observe(agent)
#             if isinstance(obs, dict) and "observation" in obs:
#                 obs = obs["observation"]
#             for _ in range(self.k):
#                 self.stacks[agent].append(obs)

#     def observe(self, agent: AgentID) -> np.ndarray:
#         obs = self.env.observe(agent)
#         if isinstance(obs, dict) and "observation" in obs:
#             obs = obs["observation"]
#         self.stacks[agent].append(obs)
#         frames = list(self.stacks[agent])[::-1]
#         return np.concatenate(frames, axis=0 if self.channel_first else -1)

#     def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
#         orig_space = self.env.observation_space(agent)
#         obs_space = (
#             orig_space["observation"]
#             if isinstance(orig_space, gymnasium.spaces.Dict)
#             else orig_space
#         )
#         shape = obs_space.shape
#         if len(shape) == 3:
#             if self.channel_first:
#                 c, h, w = shape
#                 new_shape = (c * self.k, h, w)
#             else:
#                 h, w, c = shape
#                 new_shape = (h, w, c * self.k)
#         elif len(shape) == 1:
#             new_shape = (shape[0] * self.k,)
#         else:
#             raise NotImplementedError
#         return gymnasium.spaces.Box(
#             low=np.min(obs_space.low),
#             high=np.max(obs_space.high),
#             shape=new_shape,
#             dtype=obs_space.dtype,
#         )

#     def step(self, action: ActionType) -> None:
#         self.env.step(action)

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         if observe and not term and not trunc:
#             obs = self.observe(self.env.agent_selection)
#         else:
#             obs = None
#         return obs, reward, term, trunc, info

#     def state(self) -> np.ndarray:
#         return self.env.state()
