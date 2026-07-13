from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and variance with the parallel variance algorithm.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


def _copy_info_with_final_observation(
    info: Dict[str, Any], transform: Callable[[np.ndarray], np.ndarray]
) -> Dict[str, Any]:
    """
    Applies an observation transform to Gymnasium vector final observations.

    Gymnasium vector envs return reset observations from step() after autoreset, and
    stash true terminal/time-limit observations in info["final_observation"]. Value
    bootstrapping must see those observations in the same space as normal obs.
    """
    if not isinstance(info, dict) or "final_observation" not in info:
        return info

    mask = info.get("_final_observation")
    if mask is None:
        return info

    final_observations = list(info["final_observation"])
    for env_idx, has_final_observation in enumerate(mask):
        if has_final_observation and final_observations[env_idx] is not None:
            final_observations[env_idx] = transform(final_observations[env_idx])

    copied = dict(info)
    copied["final_observation"] = final_observations
    return copied


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalizes single-environment observations with running mean and variance.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.obs_rms.update(np.asarray(observation)[None])
        return self.normalize_observation(observation)

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )


class VecEnvWrapper:
    """
    Minimal Gymnasium vector-env wrapper base.

    Gymnasium's vector wrapper API has changed across releases; this small adapter
    keeps the wrappers here usable with standard vector envs and Puffer-like envs.
    """

    def __init__(self, venv: Any):
        self.venv = venv
        self.env = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.single_observation_space = getattr(
            venv, "single_observation_space", venv.observation_space
        )
        self.single_action_space = getattr(
            venv, "single_action_space", venv.action_space
        )
        self.metadata = getattr(venv, "metadata", {})
        self.render_mode = getattr(venv, "render_mode", None)

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step(self, actions):
        return self.venv.step(actions)

    def render(self):
        return self.venv.render()

    def close(self):
        return self.venv.close()

    def __getattr__(self, name: str):
        return getattr(self.venv, name)


class VecTransformObservation(VecEnvWrapper):
    """
    Applies a stateless observation transform to vector observations.
    """

    def __init__(
        self,
        venv: Any,
        transform: Callable[[np.ndarray], np.ndarray],
        observation_space: Optional[gym.Space] = None,
    ):
        super().__init__(venv)
        self.transform = transform
        if observation_space is not None:
            self.observation_space = observation_space
            self.single_observation_space = observation_space

    def _transform_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return _copy_info_with_final_observation(info, self.transform)

    def reset(self, **kwargs):
        obs, info = self.venv.reset(**kwargs)
        return self.transform(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.venv.step(actions)
        return (
            self.transform(obs),
            rewards,
            terminated,
            truncated,
            self._transform_info(info),
        )


class VecTransformReward(VecEnvWrapper):
    """
    Applies a stateless reward transform to vector rewards.
    """

    def __init__(self, venv: Any, transform: Callable[[np.ndarray], np.ndarray]):
        super().__init__(venv)
        self.transform = transform

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.venv.step(actions)
        return obs, self.transform(rewards), terminated, truncated, info


class VecNormalize(VecEnvWrapper):
    """
    SB3-style vector normalization for observations and discounted returns.

    Observation normalization also transforms Gymnasium final observations in info,
    which is required for correct value bootstrapping at time limits.
    """

    def __init__(
        self,
        venv: Any,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        training: bool = True,
    ):
        super().__init__(venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

        self.obs_rms = (
            RunningMeanStd(shape=self.single_observation_space.shape)
            if norm_obs
            else None
        )
        self.ret_rms = RunningMeanStd(shape=()) if norm_reward else None
        self.returns = np.zeros(self.num_envs, dtype=np.float64)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs
        normalized = (obs - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)

    def normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        if self.ret_rms is None:
            return rewards
        normalized = rewards / np.sqrt(self.ret_rms.var + self.epsilon)
        return np.clip(normalized, -self.clip_reward, self.clip_reward).astype(
            np.float32
        )

    def reset(self, **kwargs):
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        obs, info = self.venv.reset(**kwargs)
        if self.obs_rms is not None and self.training:
            self.obs_rms.update(obs)
        return self.normalize_obs(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.venv.step(actions)

        if self.obs_rms is not None and self.training:
            self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)
        info = _copy_info_with_final_observation(info, self.normalize_obs)

        self.returns = self.returns * self.gamma + rewards
        if self.ret_rms is not None and self.training:
            self.ret_rms.update(self.returns)
        rewards = self.normalize_reward(rewards)

        dones = np.logical_or(terminated, truncated)
        self.returns[dones] = 0.0
        return obs, rewards, terminated, truncated, info


class VecNormalizeObservation(VecNormalize):
    def __init__(self, venv: Any, **kwargs):
        super().__init__(venv, norm_obs=True, norm_reward=False, **kwargs)


class VecNormalizeReward(VecNormalize):
    def __init__(self, venv: Any, **kwargs):
        super().__init__(venv, norm_obs=False, norm_reward=True, **kwargs)
