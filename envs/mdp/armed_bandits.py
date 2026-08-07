import numpy as np

import gymnasium as gym
from gymnasium import spaces


class ArmedBanditsEnv(gym.Env):
    metadata = {"render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        means=[np.random.uniform(-1, 1) for _ in range(1000)],
        std_devs=[np.random.rand() for _ in range(1000)],
        steps=1000,
    ):
        assert len(means) == len(std_devs)
        self.means = means
        self.std_devs = std_devs
        self.observation_space = spaces.Discrete(1)  # or 0
        self.action_space = spaces.Discrete(len(means))

        self.total_steps = steps
        self.spec.reward_threshold = np.max(means) * self.total_steps

        assert render_mode is None

    def _get_obs(self):
        return 0

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)  # might not set tfp seed

        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = np.random.normal(self.means[action], self.std_devs[action])
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, False, self.current_step == self.total_steps, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
