import gymnasium as gym
import numpy as np

from atomic_rl.envs.wrappers.normalization import VecNormalize, VecTransformObservation
from atomic_rl.envs.wrappers.pomdp import VecFlickeringObservation


class DummyVectorEnv:
    def __init__(self):
        self.num_envs = 2
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, 2), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.MultiDiscrete([2, 2])
        self.metadata = {}
        self.render_mode = None

    def reset(self, **kwargs):
        obs = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        return obs, {}

    def step(self, actions):
        obs = np.array([[3.0, 5.0], [1.0, 2.0]], dtype=np.float32)
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        terminated = np.array([False, False])
        truncated = np.array([False, True])
        info = {
            "final_observation": [
                None,
                np.array([5.0, 8.0], dtype=np.float32),
            ],
            "_final_observation": np.array([False, True]),
        }
        return obs, rewards, terminated, truncated, info

    def close(self):
        pass


def test_vec_normalize_transforms_final_observation_without_stat_update():
    env = VecNormalize(
        DummyVectorEnv(), norm_obs=True, norm_reward=False, training=False
    )
    env.obs_rms.mean = np.array([1.0, 2.0])
    env.obs_rms.var = np.array([4.0, 9.0])

    obs, rewards, terminated, truncated, info = env.step(np.array([0, 1]))

    np.testing.assert_allclose(obs, np.array([[1.0, 1.0], [0.0, 0.0]]), atol=1e-6)
    np.testing.assert_allclose(
        info["final_observation"][1], np.array([2.0, 2.0]), atol=1e-6
    )
    assert info["final_observation"][0] is None
    np.testing.assert_array_equal(rewards, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(terminated, np.array([False, False]))
    np.testing.assert_array_equal(truncated, np.array([False, True]))


def test_vec_transform_observation_transforms_final_observation():
    env = VecTransformObservation(DummyVectorEnv(), lambda obs: obs + 1.0)

    obs, _, _, _, info = env.step(np.array([0, 1]))

    np.testing.assert_allclose(obs, np.array([[4.0, 6.0], [2.0, 3.0]]))
    np.testing.assert_allclose(info["final_observation"][1], np.array([6.0, 9.0]))
    assert info["final_observation"][0] is None


def test_vec_flickering_observation_transforms_final_observation():
    env = VecFlickeringObservation(DummyVectorEnv(), prob=1.0)

    obs, _, _, _, info = env.step(np.array([0, 1]))

    np.testing.assert_allclose(obs, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(
        info["final_observation"][1], np.zeros(2, dtype=np.float32)
    )
    assert info["final_observation"][0] is None
