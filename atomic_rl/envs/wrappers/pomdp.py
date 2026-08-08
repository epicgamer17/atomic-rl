import numpy as np
import gymnasium as gym

from atomic_rl.envs.wrappers.normalization import VecTransformObservation


class FlickeringObservation(gym.ObservationWrapper):
    """
    With probability `prob`, the observation is completely obscured (returns all zeros).
    This replicates the flickering POMDP environments used in the DRQN paper.

    Models with an LSTM should perform better on flickering observations than non-LSTM models
    (ie train on MDP, transfer to POMDP version and do better).

    TODO: should the non-POMDP version still use framestacking? the LSTM version does not.
    """

    def __init__(self, env: gym.Env, prob: float = 0.5):
        """
        Args:
            env: The environment to wrap.
            prob: The probability of obscuring the observation.
        """
        super().__init__(env)
        self.prob = prob

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Obscures the observation with probability `prob`.

        Args:
            observation: The observation to obscure.

        Returns:
            The obscured observation.
        """
        if np.random.rand() < self.prob:
            return np.zeros_like(observation)
        return observation


class VecFlickeringObservation(VecTransformObservation):
    """
    Vector-env version of FlickeringObservation.

    Each environment's observation is obscured independently. Gymnasium final
    observations are transformed through VecTransformObservation as well.
    """

    def __init__(
        self,
        venv,
        prob: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        self.prob = prob
        self.rng = rng if rng is not None else np.random.default_rng()
        super().__init__(venv, self._flicker)

    def _flicker(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations)
        if observations.ndim == len(self.single_observation_space.shape):
            if self.rng.random() < self.prob:
                return np.zeros_like(observations)
            return observations

        mask_shape = (observations.shape[0],) + (1,) * (observations.ndim - 1)
        flicker_mask = self.rng.random((observations.shape[0],)) < self.prob
        flicker_mask = flicker_mask.reshape(mask_shape)
        return np.where(flicker_mask, np.zeros_like(observations), observations)
