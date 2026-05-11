import gymnasium as gym

class FireResetEnv(gym.Wrapper):
    """
    Take the FIRE action on reset for environments that are fixed until firing.
    Used in games like Pong. (Source: Anecdotal, but standard in OpenAI Baselines)
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info
