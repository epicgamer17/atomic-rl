import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from atomic_rl.envs.wrappers.normalization import (
    WelfordNormalizeObservation,
    WelfordNormalizeReward,
)

pytestmark = pytest.mark.unit


class DummySingleEnv:
    """Minimal single-env stand-in for unit-testing the Welford wrappers."""

    def __init__(self, obs_shape=(2,)):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.metadata = {}
        self.render_mode = None
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        self.obs = np.zeros(obs_shape, dtype=np.float32)
        self.reward = 0.0
        self.terminated = False
        self.truncated = False

    def step(self, action):
        return self.obs, self.reward, self.terminated, self.truncated, {}

    def reset(self, **kwargs):
        return self.obs, {}


# ==========================================
# WelfordNormalizeObservation
# ==========================================


def test_welford_normalize_observation_uses_standard_deviation():
    """The normalization divisor is sqrt(var + eps), NOT (var + eps)."""
    env = WelfordNormalizeObservation(DummySingleEnv(obs_shape=(2,)), epsilon=1e-8)
    # Pre-seed stable stats with variance 4 (std 2), so std != variance.
    env.obs_count = torch.tensor(1000.0)
    env.obs_mean = torch.zeros(2)
    env.obs_var = torch.full((2,), 4.0)
    env.obs_sq_diff = env.obs_var * (env.obs_count - 1)

    sample = np.array([2.0, -1.0], dtype=np.float32)
    out = env.observation(sample)

    # out = (sample - mean) / sqrt(var + eps)  =>  (sample - mean) / out = sqrt(var + eps)
    recovered_divisor = (sample - env.obs_mean.numpy()) / out
    expected_divisor = np.sqrt(env.obs_var.numpy() + env.epsilon)
    np.testing.assert_allclose(recovered_divisor, expected_divisor, rtol=1e-4)


def test_welford_normalize_observation_yields_unit_variance():
    """After the stats converge, normalized observations have per-dim std ~= 1."""
    env = WelfordNormalizeObservation(DummySingleEnv(obs_shape=(2,)), epsilon=1e-8)
    rng = np.random.default_rng(0)

    for _ in range(1000):
        env.observation(rng.uniform(-1.0, 1.0, size=2).astype(np.float32))

    collected = np.stack(
        [
            env.observation(rng.uniform(-1.0, 1.0, size=2).astype(np.float32))
            for _ in range(5000)
        ]
    )
    stds = collected.std(axis=0)
    np.testing.assert_allclose(stds, np.ones(2), atol=0.1)


# ==========================================
# WelfordNormalizeReward
# ==========================================


def test_welford_normalize_reward_discounted_trace():
    """rew_u = gamma * (1 - t_mask) * rew_u + r, reset on termination/truncation."""
    raw_env = DummySingleEnv()
    env = WelfordNormalizeReward(raw_env, gamma=0.99, epsilon=1e-8)

    raw_env.reward = 2.0
    env.step(0.0)
    assert env.rew_u.item() == pytest.approx(2.0)

    raw_env.reward = 3.0
    env.step(0.0)
    assert env.rew_u.item() == pytest.approx(0.99 * 2.0 + 3.0)

    raw_env.terminated = True
    raw_env.reward = 5.0
    env.step(0.0)
    assert env.rew_u.item() == pytest.approx(5.0)

    raw_env.terminated = False
    raw_env.truncated = True
    raw_env.reward = 1.0
    env.step(0.0)
    assert env.rew_u.item() == pytest.approx(1.0)


def test_welford_normalize_reward_scales_by_sqrt_of_variance():
    """scaled_reward = r / sqrt(rew_var + eps), using the tracked variance."""
    raw_env = DummySingleEnv()
    env = WelfordNormalizeReward(raw_env, gamma=0.99, epsilon=1e-8)
    # Pre-seed stable stats so the variance is not forced to 1 (count < 2).
    env.rew_count = torch.tensor(1000.0)
    env.rew_var = torch.tensor(4.0)
    env.rew_sq_diff = env.rew_var * (env.rew_count - 1)

    raw_env.reward = 2.0
    _, scaled, _, _, _ = env.step(0.0)

    expected = 2.0 / math.sqrt(env.rew_var.item() + env.epsilon)
    assert scaled == pytest.approx(expected)


def test_welford_normalize_reward_tracks_running_mean():
    """The wrapper tracks a persistent running mean of the discounted trace.

    This matches the reference SampleMeanStd (centered variance), NOT paper
    Algorithm 5's mean-zero second moment: a running mean is maintained instead of
    hardcoding zero.
    """
    raw_env = DummySingleEnv()
    env = WelfordNormalizeReward(raw_env, gamma=0.99, epsilon=1e-8)

    raw_env.reward = 2.0
    env.step(0.0)
    assert env.rew_mean.item() == pytest.approx(2.0)  # first sample: mean = sample

    raw_env.reward = 4.0
    env.step(0.0)
    u2 = 0.99 * 2.0 + 4.0  # 5.98
    assert env.rew_u.item() == pytest.approx(u2)
    # Running mean of [2.0, 5.98] == 3.99 (mean is NOT hardcoded to zero).
    assert env.rew_mean.item() == pytest.approx(3.99, abs=1e-6)
