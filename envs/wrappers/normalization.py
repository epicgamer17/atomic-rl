from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from functional.utils import update_welford_stats


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


class WelfordNormalizeObservation(gym.ObservationWrapper):
    """
    Single-environment observation normalization using Welford's online algorithm.

    Backed by ``update_welford_stats`` from
    ``functional.utils`` so that examples share a single canonical implementation.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.device = device
        self.obs_mean = torch.zeros(*self.observation_space.shape, device=device)
        self.obs_sq_diff = torch.ones(*self.observation_space.shape, device=device)
        self.obs_var = torch.ones(*self.observation_space.shape, device=device)
        self.obs_count = torch.tensor(0.0, device=device)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.obs_mean, self.obs_sq_diff, self.obs_var, self.obs_count = (
            update_welford_stats(self.obs_mean, self.obs_sq_diff, self.obs_count, obs_t)
        )
        normalized = (obs_t - self.obs_mean.unsqueeze(0)) / torch.sqrt(
            self.obs_var.unsqueeze(0) + self.epsilon
        )

        return normalized.squeeze(0).cpu().numpy()


class WelfordNormalizeReward(gym.RewardWrapper):
    """
    Single-environment reward scaling via discounted trace + Welford (Algorithm 5).

    Tracks ``rew_u = γ·(1 - t_mask)·rew_u + r``, maintains running statistics of
    ``rew_u`` via ``update_welford_stats``, and returns ``r / σ(rew_u)``.
    The termination mask ``t_mask`` zeros the trace on ``terminated or truncated``,
    replacing the separate ``u.zero_()`` step.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.rew_u = torch.tensor(0.0, device=device)
        self.rew_sq_diff = torch.tensor(1.0, device=device)
        self.rew_var = torch.tensor(1.0, device=device)
        self.rew_count = torch.tensor(0.0, device=device)

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)

        t_mask = 1.0 if (terminated or truncated) else 0.0
        reward_t = torch.as_tensor(raw_reward, dtype=torch.float32, device=self.device)
        self.rew_u = (self.gamma * (1.0 - t_mask) * self.rew_u) + reward_t

        # NOTE: Paper Algorithm 5 (ScaleReward) hardcodes a zero mean when calling
        # SampleMeanVar: SampleMeanVar(u, 0, p, n). So this wrapper computes a
        # mean-zero second-moment scale (≈ sqrt(E[u^2])) for the discounted reward
        # trace u, NOT a centered variance.
        #
        # TODO: The authors' released code (streaming-drl normalization_wrappers.py
        # `SampleMeanStd`) instead tracks the true running mean and computes the
        # centered variance Var(u) = E[(u - mean)^2] (still scaling only, not
        # mean-centering the reward). The two differ whenever the reward trace has
        # nonzero mean (e.g. Pendulum's all-negative rewards), where E[u^2] > Var(u)
        # over-shrinks rewards. To match the reference behavior, track a persistent
        # running mean (self.rew_mean) and pass it here instead of
        # torch.zeros_like(...), e.g.:
        #   self.rew_mean, self.rew_sq_diff, self.rew_var, self.rew_count = (
        #       update_welford_stats(self.rew_mean, self.rew_sq_diff,
        #                            self.rew_count, self.rew_u.unsqueeze(0))
        #   )
        _, self.rew_sq_diff, self.rew_var, self.rew_count = update_welford_stats(
            torch.zeros_like(self.rew_u),
            self.rew_sq_diff,
            self.rew_count,
            self.rew_u.unsqueeze(0),
        )

        scaled_reward = reward_t / torch.sqrt(self.rew_var + self.epsilon)

        return obs, scaled_reward.cpu().item(), terminated, truncated, info
